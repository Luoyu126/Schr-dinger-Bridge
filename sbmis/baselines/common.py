"""Shared helpers for the continuous MIS baselines: padding a batch of graphs and checking sets."""
from __future__ import annotations

import torch


def pad_graphs(adjs, device="cpu", dtype=torch.float32):
    """Stack adjacency matrices (n_g x n_g, 0/1) into G x n x n, n = max n_g.

    Returns (A, valid): valid is G x n, 1 on real nodes. Padded nodes have no edges.
    """
    n = max(a.shape[0] for a in adjs)
    A = torch.zeros(len(adjs), n, n, dtype=dtype)
    valid = torch.zeros(len(adjs), n, dtype=dtype)
    for g, a in enumerate(adjs):
        a = torch.as_tensor(a, dtype=dtype)
        A[g, :a.shape[0], :a.shape[0]] = a
        valid[g, :a.shape[0]] = 1
    return A.to(device), valid.to(device)


def conflicts(A, m):
    """Number of set neighbours of every node: A (G, n, n), m (G, B, n) 0/1 -> (G, B, n)."""
    return torch.bmm(m, A)


def is_independent(A, m):
    """(G, B) bool: no edge inside the set m."""
    return (conflicts(A, m)*m).sum(-1) == 0


def is_maximal_independent(A, m, valid):
    """(G, B) bool: m is independent and every real node outside m has a neighbour in m."""
    k = conflicts(A, m)
    indep = (k*m).sum(-1) == 0
    outside = valid.unsqueeze(1)*(1 - m)
    covered = ((outside > 0) & (k < 0.5)).sum(-1) == 0
    return indep & covered
