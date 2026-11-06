"""Source distributions (energy prior, Dirac), the Adjoint Sampling mode and the
trainer's control-energy / stage-0 / checkpoint logging."""
import json

import numpy as np
import torch

from sbmis.asbs import AnalyticCorrector, ASBSConfig, ASBSTrainer
from sbmis.base.linear_energy_base import build_base, shifted_psd
from sbmis.energies.mis import MISEnergy
from sbmis.graphs import er_graph
from sbmis.models.nets import build_net
from sbmis.sources import build_source

F64 = torch.float64


def _energy(n=10, seed=3):
    return MISEnergy(er_graph(n, 0.3, seed=seed), 2.0, 2.0, 0.5, device="cpu", dtype=F64)


def test_energy_prior_is_tilted_gaussian():
    """grad log mu(x) = -x - 2 gamma (c + M' x)  <=>  mu ∝ N(0, I) exp(-2 gamma E'_quad)."""
    en = _energy()
    for g_rel in (1.0, 4.0, 8.0):
        src = build_source("energy_prior", en, gamma_rel=g_rel)
        gamma = src.info["gamma"]
        Ms = torch.tensor(shifted_psd(en.M64.numpy()))
        assert abs(gamma - g_rel / torch.linalg.eigvalsh(Ms).max().item()) < 1e-12
        x = torch.randn(16, en.n, dtype=F64)
        score = -(x - src.mean64) @ torch.linalg.inv(src.cov64)
        want = -x - 2 * gamma * (en.c64 + x @ Ms)
        assert (score - want).abs().max() < 1e-8
        assert torch.linalg.eigvalsh(src.cov64).min() > 0


def test_energy_prior_sampling_moments():
    torch.manual_seed(0)
    src = build_source("energy_prior", _energy(), gamma_rel=4.0)
    x = src.sample(200_000, "cpu").double()
    assert (x.mean(0) - src.mean64).abs().max() < 0.01
    assert (torch.cov(x.T) - src.cov64).abs().max() < 0.01


def test_delta_source_and_analytic_corrector():
    en = _energy()
    src = build_source("delta", en)
    assert (src.sample(5, "cpu") == 0).all()
    base = build_base("ve_geometric", en, sigma=1.0, sigma_min=0.001, dtype=F64)
    h = AnalyticCorrector(base)
    x1 = torch.randn(7, en.n, dtype=F64)
    # p_1^base = N(0, sigma_max^2 - sigma_min^2) for X_0 = 0
    assert (h(torch.ones(7, 1), x1) + x1 / base.total_var).abs().max() < 1e-10


def _tiny_run(tmp_path, base_kind, source_kind, analytic=False, eval_every=1, n_eval=64,
              **bkw):
    torch.manual_seed(0)
    en = MISEnergy(er_graph(8, 0.3, seed=1), 2.0, 2.0, 0.5, device="cpu")
    base = build_base(base_kind, en, sigma=1.0, device="cpu", **bkw)
    src = build_source(source_kind, en, gamma_rel=4.0)
    adj = en.A64.numpy()
    u = build_net("mlp", adj, hidden=32, layers=2)
    h = AnalyticCorrector(base) if analytic else build_net("mlp", adj, hidden=32, layers=2)
    cfg = ASBSConfig(n_train_steps=10, resample_size=64, sample_batch=64, train_batch=64,
                     itr_per_epoch=2, am_epochs=2, cm_epochs=2, max_stages=2,
                     fixed_corrector=analytic, save_at=(128, 10**9), kl_samples=64,
                     eval_every_stages=eval_every)
    evals = lambda tr: {"x1_mean": float(tr.sample(n_eval, 10).mean())}
    tr = ASBSTrainer(en, base, u, h, cfg, device="cpu", out_dir=tmp_path, eval_fn=evals,
                     log=lambda *_: None, source=src)
    summ = tr.run()
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    return summ, recs


def test_trainer_logs_and_checkpoints(tmp_path):
    summ, recs = _tiny_run(tmp_path, "linear_energy_shift", "energy_prior", rho=4.0)
    assert summ["status"] == "ok"
    assert recs[0]["stage"] == 0 and recs[0]["grad_calls"] == 0 and "x1_mean" in recs[0]
    assert recs[0]["kl_ref"] == 0.0          # u = 0 at initialisation
    assert all(r["ctrl_energy_stage"] >= 0 for r in recs[1:])
    assert recs[-1]["kl_ref"] > 0 and recs[-1]["ctrl_energy_stage"] > 0
    assert (tmp_path / "ckpt_128.pt").exists() and not (tmp_path / "ckpt_1000000000.pt").exists()


def test_adjoint_sampling_mode(tmp_path):
    summ, recs = _tiny_run(tmp_path, "ve_geometric", "delta", analytic=True, sigma_min=0.001)
    assert summ["status"] == "ok"
    assert all(r["cm_loss"] is None for r in recs[1:])
    # AM-only stages: 2 epochs x 64 samples per stage
    assert [r["grad_calls"] for r in recs] == [0, 128, 256]


def test_evaluation_does_not_change_training(tmp_path):
    """Evaluations run on a forked RNG: the training losses do not depend on the schedule."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _tiny_run(tmp_path / "a", "linear_energy_shift", "gauss", rho=4.0)[1]
    b = _tiny_run(tmp_path / "b", "linear_energy_shift", "gauss", rho=4.0, eval_every=2,
                  n_eval=128)[1]
    la = [(r["am_loss"], r["cm_loss"]) for r in a[1:]]
    lb = [(r["am_loss"], r["cm_loss"]) for r in b[1:]]
    assert la == lb
