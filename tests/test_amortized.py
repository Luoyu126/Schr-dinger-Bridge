"""Amortized pieces against their single-graph counterparts.

Every batched quantity (energy, gradient, each reference process's transition / bridge /
lean adjoint / CM target / integrator step, energy-prior source) must equal the single-graph
implementation sample by sample, in original coordinates, both for training minibatches
(one graph per sample) and for grouped simulation batches (K samples per graph). Pad nodes
must not influence real nodes.
"""
import numpy as np
import pytest
import torch

from sbmis.amortized import (AmortizedConfig, AmortizedTrainer, AnalyticCorrectorAmortized,
                             GraphPool, ProblemSpec, decode_metrics)
from sbmis.base.linear_energy_base import (LinearEnergyBase, build_base, random_rotation,
                                           shifted_psd)
from sbmis.datasets import GraphSet
from sbmis.energies.mis import MISEnergy
from sbmis.models.difusco_gnn import DifuscoGNN
from sbmis.sources import energy_prior

F64 = torch.float64


def _gset():
    return GraphSet.generate_er(9, 14, 0.3, count=6, base_seed=123)


def _pool(**kw):
    return GraphPool(_gset(), ProblemSpec(**kw), device="cpu")


def _single(pool, gi):
    sp = pool.spec
    en = MISEnergy(pool.g.adj(gi), sp.lambda_p, sp.kappa_w, sp.temp, device="cpu", dtype=F64)
    if sp.base == "linear_energy_shift_rot":
        R = random_rotation(en.n, 1_000_003 * sp.rot_seed + gi)
        Ms = shifted_psd(en.M64.numpy())
        base = LinearEnergyBase(R @ Ms @ R.T, R @ en.c64.numpy(), sigma=sp.sigma, rho=sp.rho,
                                dtype=F64)
    else:
        base = build_base(sp.base, en, sigma=sp.sigma, rho=sp.rho, dtype=F64,
                          iso_match_rho=sp.iso_match_rho, iso_match=sp.iso_match)
    return en, base


def test_batched_energy_matches_single():
    pool = _pool()
    gids = np.array([0, 3, 3, 5])
    b = pool.batch(gids)
    x = torch.randn(b.S, pool.n_max, dtype=F64) * b.mask
    e, g = pool.e_tgt(x, b), pool.grad_u(x, b)
    for s, gi in enumerate(gids):
        en, _ = _single(pool, gi)
        n = en.n
        assert abs(float(e[s]) - float(en.e_tgt(x[s:s + 1, :n]))) < 1e-9
        assert (g[s, :n] - en.grad_u(x[s:s + 1, :n], count=False)[0]).abs().max() < 1e-9
        assert (g[s, n:] == 0).all()


KINDS = [dict(base="zero"), dict(base="vp"), dict(base="linear_energy", rho=2.0),
         dict(base="linear_energy", rho=-2.0), dict(base="linear_energy_shift", rho=4.0),
         dict(base="linear_energy_shift_rot", rho=4.0, rot_seed=3),
         dict(base="iso_ou", iso_match_rho=4.0), dict(base="iso_ou", iso_match_rho=8.0,
                                                      iso_match="m2")]


@pytest.mark.parametrize("kw", KINDS, ids=lambda k: "-".join(str(v) for v in k.values()))
@pytest.mark.parametrize("grouped", [1, 3])
def test_batched_base_matches_single(kw, grouped):
    torch.manual_seed(0)
    pool = _pool(**kw)
    gl = np.array([1, 4]) if grouped > 1 else np.array([1, 4, 4, 2])
    gids = np.repeat(gl, grouped)
    b = pool.batch(gids)
    base = pool.base_ctx(gl, grouped=grouped)
    S, nm = b.S, pool.n_max
    m = b.mask.to(torch.float32)
    x0, x1 = torch.randn(S, nm) * m, torch.randn(S, nm) * m
    z, v, a1 = torch.randn(S, nm) * m, torch.randn(S, nm) * m, torch.randn(S, nm) * m
    t = torch.rand(S, 1)
    out = {"bridge": base.bridge_sample(t, x0, x1, z=z) * m,
           "lean": base.lean_adjoint(a1, t) * m,
           "cm": base.cm_target(x0, x1) * m,
           "step": base.step(x0, 0.3, 0.04, v=v, z=z) * m}
    for s, gi in enumerate(gids):
        _, sb = _single(pool, gi)
        n = int(pool.g.n[gi])
        f = lambda a: a[s:s + 1, :n].double()
        want = {"bridge": sb.bridge_sample(t[s:s + 1].double(), f(x0), f(x1), z=f(z)),
                "lean": sb.lean_adjoint(f(a1), t[s:s + 1].double()),
                "cm": sb.cm_target(f(x0), f(x1)),
                "step": sb.step(f(x0), 0.3, 0.04, v=f(v), z=f(z))}
        for k in out:
            assert (out[k][s, :n].double() - want[k][0]).abs().max() < 2e-4, (k, s)
            assert (out[k][s, n:] == 0).all()


def test_energy_prior_source_matches_single():
    pool = _pool(source="energy_prior", prior_gamma=4.0)
    torch.manual_seed(1)
    gl, K = np.array([2, 5]), 40000
    b = pool.batch(np.repeat(gl, K))
    x = pool.sample_source(gl, K, b.mask.float())
    for k, gi in enumerate(gl):
        en, _ = _single(pool, gi)
        ref = energy_prior(en, 4.0)
        n = en.n
        xs = x[k * K:(k + 1) * K, :n].double()
        assert (xs.mean(0) - ref.mean64).abs().max() < 0.02
        assert (torch.cov(xs.T) - ref.cov64).abs().max() < 0.02
        assert (x[k * K:(k + 1) * K, n:] == 0).all()


def test_pads_do_not_affect_real_nodes():
    torch.manual_seed(2)
    pool = _pool()
    net = DifuscoGNN(hidden=16, n_layers=2)
    for mod in net.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=0.3)
    b = pool.batch(np.array([0, 1, 2]))
    x = torch.randn(b.S, pool.n_max) * b.mask
    t = torch.rand(b.S)
    y1 = net(t, x.reshape(-1), b).reshape(b.S, -1)
    x2 = x + torch.randn_like(x) * (~b.mask)  # change pad values only
    y2 = net(t, x2.reshape(-1), b).reshape(b.S, -1)
    assert ((y1 - y2) * b.mask).abs().max() < 1e-6


def _tiny_trainer(tmp_path, analytic=False, **kw):
    torch.manual_seed(0)
    pool = _pool(**kw)
    u = DifuscoGNN(hidden=16, n_layers=2)
    h = AnalyticCorrectorAmortized(pool) if analytic else DifuscoGNN(hidden=16, n_layers=2)
    cfg = AmortizedConfig(graphs_per_epoch=3, samples_per_graph=4, n_train_steps=5,
                          train_batch=8, itr_per_epoch=3, am_epochs=2, cm_epochs=2,
                          max_stages=2, amp=False, fixed_corrector=analytic, kl_graphs=2,
                          save_at=(24,))
    opt = {i: {"size": 4} for i in range(len(pool.g))}

    def evals(tr):
        b, x = tr.sample_graphs([0, 1], 4, 5)
        return decode_metrics(pool, b, x, 4, opt)
    tr = AmortizedTrainer(pool, u, h, cfg, device="cpu", out_dir=tmp_path, eval_fn=evals,
                          log=lambda *_: None)
    return tr, tr.run()


@pytest.mark.parametrize("kw", [dict(base="zero"), dict(base="linear_energy_shift", rho=4.0),
                                dict(base="zero", source="energy_prior"),
                                dict(base="iso_ou", iso_match_rho=4.0, iso_match="m2"),
                                dict(base="vp")],
                         ids=["B0", "B1p", "prior", "iso_m2", "VP"])
def test_amortized_trainer_smoke(tmp_path, kw):
    tr, summ = _tiny_trainer(tmp_path, **kw)
    assert summ["status"] == "ok" and summ["grad_calls"] == 2 * 2 * 12
    import json
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert recs[0]["stage"] == 0 and recs[0]["kl_ref"] == 0.0
    assert {"mis_size_mean", "gap_mean", "raw_feasible_frac", "kl_ref"} <= set(recs[-1])
    assert (tmp_path / "ckpt_24.pt").exists() and (tmp_path / "final.pt").exists()


@pytest.mark.parametrize("kw", [dict(base="zero"), dict(base="linear_energy_shift", rho=4.0),
                                dict(base="linear_energy_shift", rho=2.0)],
                         ids=["B0", "B1p4", "B1p2"])
def test_amortized_adjoint_sampling_mode(tmp_path, kw):
    tr, summ = _tiny_trainer(tmp_path, analytic=True, source="delta", **kw)
    assert summ["status"] == "ok" and summ["grad_calls"] == 2 * 2 * 12


@pytest.mark.parametrize("kw", [dict(base="zero"), dict(base="linear_energy_shift", rho=4.0),
                                dict(base="linear_energy", rho=2.0)],
                         ids=["B0", "B1p4", "B1_2"])
def test_analytic_corrector_is_terminal_score_for_any_base(kw):
    """With a Dirac source X0 = 0 every reference process is memoryless, so the corrector is
    grad log p_1^base(x1) with p_1^base = N(m, U diag(v) U^T) the base transition from 0 over
    [0, 1]. Checked by autodiff of that Gaussian log-density, per graph, on a padded batch."""
    torch.manual_seed(3)
    pool = _pool(source="delta", **kw)
    gids = np.array([1, 4, 4, 2])
    b = pool.batch(gids)
    m = b.mask.to(torch.float32)
    x1 = torch.randn(b.S, pool.n_max) * 1.5 * m
    h = AnalyticCorrectorAmortized(pool)(torch.ones(b.S), x1.reshape(-1), b).reshape(b.S, -1)
    for s, gi in enumerate(gids):
        _, sb = _single(pool, gi)
        n = int(pool.g.n[gi])
        mean, var = sb.transition_mean_var(torch.zeros(1, n, dtype=F64), 1.0)
        U = torch.eye(n, dtype=F64) if sb.identity else sb.U64
        cov = U @ torch.diag(var.reshape(-1).expand(n)) @ U.T
        x = x1[s:s + 1, :n].double().requires_grad_(True)
        logp = torch.distributions.MultivariateNormal(mean[0], covariance_matrix=cov).log_prob(x)
        want, = torch.autograd.grad(logp.sum(), x)
        assert (h[s, :n].double() - want[0]).abs().max() / want.abs().max() < 2e-4, s
        assert (h[s, n:] == 0).all()


@pytest.mark.parametrize("base", ["zero", "vp", "linear_energy_shift", "iso_ou"])
@pytest.mark.parametrize("source", ["gauss", "energy_prior"])
def test_cache_eviction_with_more_graphs_than_slots(base, source):
    """The LRU cache must evict cleanly (and recompute identically) for every base/source."""
    kw = dict(base=base, source=source, rho=4.0)
    if base == "iso_ou":
        if source != "gauss":
            return
        kw["iso_match_rho"] = 4.0
    pool = GraphPool(_gset(), ProblemSpec(**kw), device="cpu", cache_size=2)
    gl = np.arange(6)
    m = pool.batch(np.repeat(gl, 2)).mask.float()
    first = pool.base_ctx(gl[:1])
    for _ in range(2):
        for g in gl:
            pool.base_ctx(np.array([g]))
            pool.sample_source(np.array([g]), 2, m[:2])
    again = pool.base_ctx(gl[:1])
    if hasattr(first, "theta"):
        assert torch.equal(first.theta, again.theta) and torch.equal(first.b, again.b)
    assert len(pool.cache) <= 2


def test_decode_metrics_best_of_matches_single_instance_definition():
    from sbmis.evaluate import Decoder, mis_metrics
    pool = _pool()
    torch.manual_seed(5)
    K = 32
    b = pool.batch(np.repeat(np.array([3]), K))
    x = torch.randn(K, pool.n_max) * b.mask
    n = int(pool.g.n[3])
    en, _ = _single(pool, 3)
    ref = mis_metrics(x[:, :n], en, Decoder(pool.g.adj(3), device="cpu"))
    got = decode_metrics(pool, b, x, K, None)
    assert abs(got["best_of_16"] - ref["best_of_16"]) < 1e-6
    assert abs(got["mis_size_mean"] - ref["mis_size_mean"]) < 1e-6
