"""Checks of the pCQO and PQQA reimplementations (sbmis/baselines) on small graphs, on CPU."""
import itertools

import numpy as np
import torch

from sbmis.baselines.common import is_independent, is_maximal_independent, pad_graphs
from sbmis.baselines.pcqo import pcqo
from sbmis.baselines.pqqa import pqqa
from sbmis.graphs import er_graph


def brute_mis(adj):
    n = adj.shape[0]
    best = 0
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits)
        if x @ adj @ x == 0:
            best = max(best, x.sum())
    return best


def test_maximal_check_matches_cpp_formula():
    """clamp(m - 0.1(-1 + m(nA - A')), 0, 1) == m  <=>  maximal independent set."""
    torch.manual_seed(0)
    adjs = [er_graph(n, 0.3, seed=n) for n in (9, 12)]
    A, valid = pad_graphs(adjs)
    m = (torch.rand(2, 4000, A.shape[1]) < 0.3).float()*valid.unsqueeze(1)
    ours = is_maximal_independent(A, m, valid)
    for g, a in enumerate(adjs):
        n = a.shape[0]
        a = torch.as_tensor(a, dtype=torch.float32)
        upd = n*a - (1 - a - torch.eye(n))
        mm = m[g, :, :n]
        cpp = ((mm - 0.1*(-1 + mm@upd)).clamp(0, 1) == mm).all(1)
        assert torch.equal(cpp, ours[g])
        assert cpp.any() and (~cpp).any()


def test_pcqo_returns_maximal_sets():
    adjs = [er_graph(n, 0.3, seed=n) for n in (10, 13, 14)]
    out = pcqo(adjs, lr=1e-3, steps=600, steps_per_batch=30, gamma=20, gamma_prime=1, batch=32,
               dtype=torch.float32, device="cpu", seed=0)
    for g, a in enumerate(adjs):
        s = out["best_set"][g].float()
        A, valid = pad_graphs([a])
        assert is_maximal_independent(A, s.view(1, 1, -1), valid).item()
        assert s.sum().item() == out["best_size"][g].item()
        assert s.sum().item() == brute_mis(a)                              # tiny graphs: optimum found
    assert len(out["curve"]) == 600//30


def test_pqqa_feasible_and_optimal_on_tiny_graphs():
    adjs = [er_graph(n, 0.3, seed=n) for n in (10, 13, 14)]
    out = pqqa(adjs, runs=32, steps=300, device="cpu", seed=0, record_every=50)
    for g, a in enumerate(adjs):
        s = out["best_set"][g].float()
        A, _ = pad_graphs([a])
        assert is_independent(A, s.view(1, 1, -1)).item()
        assert s.sum().item() == out["best_feasible"][g].item() == brute_mis(a)
    assert [c["step"] for c in out["curve"]] == [0, 50, 100, 150, 200, 250, 300]

