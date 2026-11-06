"""Graph instances and exact MIS references.

Graphs are generated deterministically from a seed (no data files needed).
ER edge probability follows DIFUSCO's ER generator (p = 0.15).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def er_graph(n: int, p: float = 0.15, seed: int = 0, connected: bool = True) -> np.ndarray:
    """Adjacency (n, n) float64 of an ER graph; resamples until connected if asked."""
    s = seed
    while True:
        G = nx.gnp_random_graph(n, p, seed=s)
        if not connected or nx.is_connected(G):
            break
        s += 10_000
    return nx.to_numpy_array(G, nodelist=range(n), dtype=np.float64)


# Small graphs for U1 (n <= 16). Fixed (n, p, seed) so they are reproducible.
U1_GRAPHS = {
    "g0": dict(n=12, p=0.3, seed=0),
    "g1": dict(n=16, p=0.3, seed=1),
    "g2": dict(n=14, p=0.3, seed=2),
}


def get_graph(spec: str) -> np.ndarray:
    """spec: 'g0' | 'g1' | 'g2' | 'er:<n>:<p>:<seed>'."""
    if spec in U1_GRAPHS:
        return er_graph(**U1_GRAPHS[spec])
    if spec.startswith("er:"):
        _, n, p, seed = spec.split(":")
        return er_graph(int(n), float(p), int(seed))
    raise ValueError(spec)


def is_independent(adj: np.ndarray, s: np.ndarray) -> bool:
    s = np.asarray(s, dtype=np.float64)
    return float(s @ adj @ s) == 0.0


def all_maximum_independent_sets(adj: np.ndarray) -> list[np.ndarray]:
    """Brute force over all subsets (n <= 20)."""
    n = adj.shape[0]
    assert n <= 20
    best, sets = -1, []
    nbr = [set(np.nonzero(adj[i])[0]) for i in range(n)]
    # enumerate maximal independent sets via networkx cliques of the complement
    Gc = nx.complement(nx.from_numpy_array(adj))
    for cl in nx.find_cliques(Gc):
        k = len(cl)
        if k > best:
            best, sets = k, [cl]
        elif k == best:
            sets.append(cl)
    out = []
    for cl in sets:
        v = np.zeros(n, dtype=np.int64)
        v[cl] = 1
        assert all(not (nbr[i] & set(cl)) for i in cl)
        out.append(v)
    return out


def exact_mis_size(adj: np.ndarray, time_limit: float = 600.0) -> tuple[int, np.ndarray, bool]:
    """Exact MIS via MILP (scipy HiGHS). Returns (size, indicator, proven_optimal)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = adj.shape[0]
    iu = np.argwhere(np.triu(adj, 1) > 0)
    rows = []
    # edge constraints, strengthened with maximal cliques (valid inequalities)
    G = nx.from_numpy_array(adj)
    for cl in nx.find_cliques(G):
        if len(cl) >= 2:
            r = np.zeros(n)
            r[cl] = 1
            rows.append(r)
    if not rows:
        return n, np.ones(n, dtype=np.int64), True
    Acon = np.stack(rows)
    res = milp(
        c=-np.ones(n),
        constraints=LinearConstraint(Acon, -np.inf, 1.0),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "disp": False},
    )
    x = np.round(res.x).astype(np.int64)
    assert is_independent(adj, x)
    assert len(iu) == 0 or np.all(x[iu[:, 0]] + x[iu[:, 1]] <= 1)
    return int(x.sum()), x, bool(res.status == 0)
