"""MALA refinement of the AM buffer (AmortizedConfig.mcmc_steps)."""
import json

import numpy as np
import pytest
import torch

from sbmis.amortized import (AmortizedConfig, AmortizedTrainer, AnalyticCorrectorAmortized,
                             GraphPool, ProblemSpec)
from sbmis.datasets import GraphSet
from sbmis.energies.mis import p_and_dp
from sbmis.models.difusco_gnn import DifuscoGNN


def _trainer(gset=None, tmp_path=None, spec=None, **cfg_kw):
    torch.manual_seed(0)
    gset = gset or GraphSet.generate_er(9, 14, 0.3, count=6, base_seed=123)
    pool = GraphPool(gset, ProblemSpec(**({"source": "delta", "pmap": "smooth", "temp": 0.1,
                                           "kappa_w": 1.0} | (spec or {}))), device="cpu")
    cfg = AmortizedConfig(**({"graphs_per_epoch": 3, "samples_per_graph": 4, "n_train_steps": 5,
                              "train_batch": 8, "itr_per_epoch": 3, "am_epochs": 2,
                              "max_stages": 2, "amp": False, "fixed_corrector": True,
                              "clip_mode": "node", "alpha_max": 5.0} | cfg_kw))
    return AmortizedTrainer(pool, DifuscoGNN(hidden=16, n_layers=2), AnalyticCorrectorAmortized(pool),
                            cfg, device="cpu", out_dir=tmp_path, log=lambda *_: None)


def test_refinement_leaves_the_target_invariant():
    """Isolated nodes: nu_T factorizes into exp(-(-p(x) + (x^2 - 1)^2) / T) per node."""
    k, n, T = 2000, 3, 1.0
    gset = GraphSet(np.full(1, n, dtype=np.int32), np.zeros(2, dtype=np.int64),
                    np.zeros((0, 2), dtype=np.int32))
    tr = _trainer(gset, spec={"temp": T}, mcmc_steps=1500)
    b = tr.pool.batch(np.zeros(k, dtype=int))
    x = tr._refine(torch.randn(k, tr.pool.n_max, dtype=torch.float64), b, T)
    grid = torch.linspace(-4, 4, 40001, dtype=torch.float64)
    w = torch.exp(-(-p_and_dp(grid, "smooth")[0] + (grid ** 2 - 1) ** 2) / T)
    w = w / w.sum()
    for f, tol in ((lambda z: z, 0.04), (lambda z: z ** 2, 0.05)):
        assert abs(float(f(x).mean() - (w * f(grid)).sum())) < tol
    assert 0.45 < tr._mcmc_stats["mcmc_accept"] < 0.7
    assert tr.mcmc_grad_calls == k * 1500


def test_refinement_leaves_the_all_off_corner_and_keeps_pads_zero():
    tr = _trainer(mcmc_steps=300)
    b = tr.pool.batch(np.array([0, 1, 2, 3, 4, 5]))
    x = -1.0 * b.mask.float()
    y = tr._refine(x, b, 0.5)
    assert (y[~b.mask] == 0).all()
    assert tr._mcmc_stats["raw1_sampler"] == 0.0 and tr._mcmc_stats["raw1_refined"] > 1.0


def test_min_temp_gates_refinement_and_buffer_gradient_is_at_refined_x1():
    tr = _trainer(mcmc_steps=5, mcmc_min_temp=0.2)  # T = 0.1 < 0.2: off
    tr.populate_am(init=False)
    assert tr.mcmc_grad_calls == 0 and tr._mcmc_stats is None
    tr = _trainer(mcmc_steps=5, mcmc_min_temp=0.05)
    tr.populate_am(init=False)
    d = tr.buf_am.data
    S = len(d["gid"])
    assert tr.mcmc_grad_calls == S * 5 and tr.grad_calls == S
    b = tr.pool.batch(d["gid"].numpy())
    assert torch.equal(d["ge"], tr.pool.grad_e(d["x1"], b))


def test_refinement_needs_dirac_source_and_no_binarization():
    with pytest.raises(ValueError):
        _trainer(spec={"source": "gauss"}, mcmc_steps=5, fixed_corrector=False)
    with pytest.raises(ValueError):
        _trainer(mcmc_steps=5, temp_start=1.0, bin_gamma_start=-2.0, anneal_epochs=2)


def test_refined_run_logs_statistics(tmp_path):
    tr = _trainer(tmp_path=tmp_path, mcmc_steps=3, temp_start=0.5, anneal_epochs=2,
                  mcmc_min_temp=0.25)
    summ = tr.run()
    assert summ["status"] == "ok" and summ["mcmc_grad_calls"] == tr.mcmc_grad_calls > 0
    ep = [json.loads(l) for l in (tmp_path / "epochs.jsonl").read_text().splitlines()]
    # T_e = 0.5, 0.3, then 0.1 < mcmc_min_temp = 0.25: refinement in the first two AM epochs only
    assert ["mcmc_accept" in e for e in ep] == [True, True, False, False]
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["mcmc_epochs"] for r in recs] == [2, 0]
    assert recs[0]["raw1_refined"] is not None and recs[1]["raw1_refined"] is None
