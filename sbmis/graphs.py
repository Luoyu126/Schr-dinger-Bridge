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


def local_search_mis(adj: np.ndarray, restarts: int = 20, iters: int = 20_000,
                     seed: int = 0) -> np.ndarray:
    """Iterated local search with (1,2)-swaps (ARW-style), multi-restart.

    Deterministic given `seed`. Returns the best independent set found (0/1 vector).
    """
    n = adj.shape[0]
    nbrs = [np.nonzero(adj[i])[0] for i in range(n)]
    deg = adj.sum(1)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, dtype=np.int64)

    def greedy(order):
        s = np.zeros(n, dtype=np.int64)
        tight = np.zeros(n, dtype=np.int64)  # number of solution neighbours
        for v in order:
            if s[v] == 0 and tight[v] == 0:
                s[v] = 1
                tight[nbrs[v]] += 1
        return s, tight

    def add(s, tight, v):
        s[v] = 1
        tight[nbrs[v]] += 1

    def remove(s, tight, v):
        s[v] = 0
        tight[nbrs[v]] -= 1

    for r in range(restarts):
        order = np.lexsort((rng.random(n), deg))  # min-degree greedy, random ties
        s, tight = greedy(order)
        cur_best = s.copy()
        for _ in range(iters):
            improved = False
            for v in rng.permutation(np.nonzero(s)[0]):
                # free candidates: neighbours of v whose only solution neighbour is v
                cand = [w for w in nbrs[v] if s[w] == 0 and tight[w] == 1]
                if len(cand) < 2:
                    continue
                cand = rng.permutation(cand)
                pair = None
                for i in range(len(cand)):
                    for j in range(i + 1, len(cand)):
                        if adj[cand[i], cand[j]] == 0:
                            pair = (cand[i], cand[j])
                            break
                    if pair:
                        break
                if pair:
                    remove(s, tight, v)
                    add(s, tight, pair[0])
                    add(s, tight, pair[1])
                    for w in rng.permutation(n):  # re-maximalise
                        if s[w] == 0 and tight[w] == 0:
                            add(s, tight, w)
                    improved = True
                    break
            if s.sum() > cur_best.sum():
                cur_best = s.copy()
            if not improved:
                # perturbation: force in a random non-solution vertex
                v = int(rng.choice(np.nonzero(s == 0)[0]))
                for w in nbrs[v]:
                    if s[w]:
                        remove(s, tight, w)
                add(s, tight, v)
                for w in rng.permutation(n):
                    if s[w] == 0 and tight[w] == 0:
                        add(s, tight, w)
        if cur_best.sum() > best.sum():
            best = cur_best
    assert is_independent(adj, best)
    return best


def best_known_mis(spec: str, adj: np.ndarray, cache_dir: str = "data/cache",
                   milp_time_limit: float = 1800.0) -> dict:
    """Best known MIS size for a graph, computed once and cached (shared by all runs).

    Returns {"size", "proven_optimal", "milp_size", "ls_size"}. The MILP time limit is
    wall-clock, so the cached value (not a per-run recomputation) is what makes gap
    metrics comparable across runs. A file lock serialises concurrent first calls.
    """
    import fcntl
    import json
    from pathlib import Path

    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"mis_{spec.replace(':', '_')}.json"
    with open(d / f".{f.name}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if f.exists():
            return json.loads(f.read_text())
        m_size, _, proven = exact_mis_size(adj, time_limit=milp_time_limit)
        ls = int(local_search_mis(adj).sum()) if not proven else m_size
        out = {"size": max(m_size, ls), "proven_optimal": bool(proven),
               "milp_size": m_size, "ls_size": ls, "milp_time_limit": milp_time_limit}
        f.write_text(json.dumps(out))
        return out
