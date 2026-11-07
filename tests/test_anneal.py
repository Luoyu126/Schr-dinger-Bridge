"""Training-time annealing of the AM target (AmortizedConfig.temp_start / bin_gamma_start).

The buffer keeps grad E, grad Phi and h(x1) and rebuilds the target at the current schedule;
without annealing the rebuilt target must equal the original one bit for bit.
"""
import json

import numpy as np
import pytest
import torch

from sbmis.amortized import (AmortizedConfig, AmortizedTrainer, AnalyticCorrectorAmortized,
                             GraphPool, ProblemSpec)
from sbmis.datasets import GraphSet
from sbmis.energies.mis import p_and_dp
from sbmis.models.difusco_gnn import DifuscoGNN


def _pool(**kw):
    return GraphPool(GraphSet.generate_er(9, 14, 0.3, count=6, base_seed=123), ProblemSpec(**kw),
                     device="cpu")


def _trainer(tmp_path=None, spec=None, **cfg_kw):
    torch.manual_seed(0)
    pool = _pool(**({"source": "delta", "pmap": "smooth", "temp": 0.1} | (spec or {})))
    u = DifuscoGNN(hidden=16, n_layers=2)
    cfg = AmortizedConfig(**({"graphs_per_epoch": 3, "samples_per_graph": 4, "n_train_steps": 5,
                              "train_batch": 8, "itr_per_epoch": 3, "am_epochs": 2,
                              "max_stages": 2, "amp": False, "fixed_corrector": True,
                              "kl_graphs": 2, "clip_mode": "node", "alpha_max": 5.0} | cfg_kw))
    return AmortizedTrainer(pool, u, AnalyticCorrectorAmortized(pool), cfg, device="cpu",
                            out_dir=tmp_path, log=lambda *_: None)


@pytest.mark.parametrize("pmap", ["lin", "clamp", "smooth"])
def test_grad_bin_is_gradient_of_phi(pmap):
    pool = _pool(pmap=pmap)
    b = pool.batch(np.array([0, 2, 5]))
    torch.manual_seed(1)
    x = (0.9 * torch.rand(b.S, pool.n_max, dtype=torch.float64) * 2 - 0.9) * b.mask  # inside the box
    xr = x.clone().requires_grad_(True)
    p = p_and_dp(xr, pmap)[0]
    phi = ((1.0 - (2.0 * p - 1.0) ** 2) * b.mask).sum()
    want, = torch.autograd.grad(phi, xr)
    assert (pool.grad_bin(x, b) - want).abs().max() < 1e-12
    assert (pool.grad_bin(x, b)[~b.mask] == 0).all()


def test_grad_bin_pushes_off_corner_up_for_negative_gamma():
    pool = _pool(pmap="clamp")
    b = pool.batch(np.array([1]))
    x = -0.99 * torch.ones(1, pool.n_max) * b.mask
    # energy gradient of gamma * Phi at the all-off corner: gamma * grad Phi; force = -that
    force = -(-4.0) * pool.grad_bin(x, b)
    assert (force[b.mask] > 0).all()


def test_grad_u_is_grad_e_over_temp():
    pool = _pool(pmap="smooth", temp=0.075)
    b = pool.batch(np.array([0, 3]))
    x = 1.3 * torch.randn(b.S, pool.n_max) * b.mask
    assert torch.equal(pool.grad_u(x, b), pool.grad_e(x, b) / 0.075)


@pytest.mark.parametrize("clip_mode", ["node", "vector"])
def test_rebuilt_target_equals_direct_target_without_annealing(clip_mode):
    tr = _trainer(clip_mode=clip_mode)
    tr.populate_am(init=False)
    d = tr.buf_am.data
    b = tr.pool.batch(d["gid"].numpy())
    x1 = d["x1"]
    g = tr.pool.grad_u(x1, b)
    alpha = tr.cfg.alpha_max
    if clip_mode == "node":
        g = g.clamp(-alpha, alpha)
    else:
        a = alpha * torch.sqrt(b.n.float() / 100.0)
        g = g * torch.clamp(a / (torch.linalg.vector_norm(g, dim=-1) + 1e-6), max=1.0)[:, None]
    want = g + tr.h(torch.ones(b.S), x1.reshape(-1), b).reshape(b.S, -1)
    assert torch.equal(tr._terminal_target(d, b), want)
    assert "gb" not in d and tr.schedule() == (0.1, 0.0)


def test_schedule_is_linear_then_constant():
    tr = _trainer(temp_start=1.0, bin_gamma_start=-4.0, anneal_epochs=4)
    got = [tr.schedule(e) for e in (0, 1, 2, 4, 9)]
    want = [(1.0, -4.0), (0.775, -3.0), (0.55, -2.0), (0.1, 0.0), (0.1, 0.0)]
    for (T, g), (Tw, gw) in zip(got, want):
        assert abs(T - Tw) < 1e-12 and abs(g - gw) < 1e-12


def test_annealing_needs_epochs():
    with pytest.raises(ValueError):
        _trainer(temp_start=1.0)


def test_buffer_target_follows_current_schedule():
    """Samples drawn at epoch 0 must be re-targeted with the schedule of the current epoch."""
    tr = _trainer(spec={"pmap": "clamp", "well": "out"}, temp_start=1.0, bin_gamma_start=-4.0,
                  anneal_epochs=2, alpha_max=1e9)
    tr.populate_am(init=False)
    d = tr.buf_am.data
    b = tr.pool.batch(d["gid"].numpy())
    hx = tr.h(torch.ones(b.S), d["x1"].reshape(-1), b).reshape(b.S, -1)
    ge, gb = tr.pool.grad_e(d["x1"], b), tr.pool.grad_bin(d["x1"], b)
    for e in range(3):
        tr._am_epochs_done = e
        T, gam = tr.schedule()
        want = (ge + gam * gb) / T + hx
        assert torch.allclose(tr._terminal_target(d, b), want, rtol=1e-6, atol=1e-6)
    assert tr.schedule() == (0.1, 0.0)  # the true target after anneal_epochs


def test_annealed_run_logs_schedule(tmp_path):
    tr = _trainer(tmp_path, temp_start=0.5, bin_gamma_start=-2.0, anneal_epochs=3)
    summ = tr.run()
    assert summ["status"] == "ok"
    ep = [json.loads(l) for l in (tmp_path / "epochs.jsonl").read_text().splitlines()]
    assert [round(e["temp_train"], 6) for e in ep] == [0.5, round(0.5 - 0.4 / 3, 6),
                                                       round(0.5 - 0.8 / 3, 6), 0.1]
    assert ep[-1]["bin_gamma"] == 0.0
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in recs] == [1, 2]  # no eval_fn: no stage-0 record
    assert abs(recs[0]["temp_train"] - (0.5 - 0.4 / 3)) < 1e-12  # last AM epoch of stage 1
    assert recs[-1]["temp_train"] == 0.1 and recs[-1]["bin_gamma"] == 0.0
