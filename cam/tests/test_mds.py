"""Tests for envs/mds.py (minimum dominating set, DiffUCO energy). CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_mds.py

Optional: MDS_TEST_BA=<dir of BA gpickles> also checks a few real graphs.
"""
import itertools
import os
import sys
import time
import traceback
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)

import networkx as nx  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from envs.mds import Env  # noqa: E402

torch.set_num_threads(2)
A_, B_ = 1.0, 1.01


def to_data(G):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    e = [[u, v] for u, v in G.edges() if u != v]
    e = torch.tensor(e + [[v, u] for u, v in e], dtype=torch.long).T.reshape(2, -1)
    return Data(edge_index=e, edge_weight=torch.ones(e.shape[1]), num_nodes=n)


def batch_of(graphs):
    return Batch.from_data_list([to_data(G) for G in graphs])


def env(mode='continuous'):
    return Env(types.SimpleNamespace(mis_decoding=mode))


def brute_energy(G, x):
    """DiffUCO's formula with explicit products"""
    e = A_*sum(x)
    for i in G:
        prod = 1 - x[i]
        for j in G[i]:
            prod *= 1 - x[j]
        e += B_*prod
    return e


def is_dominating(G, S):
    S = set(S)
    return all(i in S or any(j in S for j in G[i]) for i in G)


def is_minimal(G, S):
    return all(not is_dominating(G, set(S) - {v}) for v in S)


def split(graph, col):
    out, ptr = [], graph.ptr.tolist()
    for i in range(len(ptr) - 1):
        out.append([j - ptr[i] for j in range(ptr[i], ptr[i + 1]) if col[j] > 0.5])
    return out


def graphs_(seed):
    return [nx.gnp_random_graph(9, 0.3, seed=seed), nx.barabasi_albert_graph(14, 2, seed=seed + 1),
            nx.gnp_random_graph(7, 0.15, seed=seed + 2)]


def test_1_energy_and_exact_flip_differences():
    graphs = graphs_(0)
    g = batch_of(graphs)
    torch.manual_seed(0)
    x = torch.randint(0, 2, (g.num_nodes, 6)).float()
    E, grad = env().energy_func(g, x, True)
    assert E.shape == (3, 6)
    err_e, err_f = 0.0, 0.0
    for k in range(6):
        for gi, G in enumerate(graphs):
            off = g.ptr[gi].item()
            xs = x[off:off + G.number_of_nodes(), k].tolist()
            err_e = max(err_e, abs(E[gi, k].item() - brute_energy(G, xs)))
            for i in G:
                xt = list(xs)
                xt[i] = 1 - xt[i]
                d = brute_energy(G, xt) - brute_energy(G, xs)
                err_f = max(err_f, abs(d - (1 - 2*xs[i])*grad[off + i, k].item()))
    assert err_e < 1e-5 and err_f < 1e-5, (err_e, err_f)
    return f'max |E - brute| {err_e:.1e}, max |flip diff - (1-2x) grad| {err_f:.1e}'


def test_2_decoders_return_minimal_dominating_sets():
    graphs = graphs_(5)
    g = batch_of(graphs)
    torch.manual_seed(1)
    sizes = {}
    for mode in ('continuous', 'discrete'):
        x = torch.rand(g.num_nodes, 5) if mode == 'continuous' else torch.randint(0, 2, (g.num_nodes, 5)).float()
        obj, sol = env(mode).evaluate(g, x.clone())
        if mode == 'continuous':                       # layouts as envs/mis.py
            sol = sol.T
            assert torch.equal(obj, sol.sum(1))
        assert sol.shape == (5, g.num_nodes)
        for k in range(5):
            for G, S in zip(graphs, split(g, sol[k])):
                assert is_dominating(G, S) and is_minimal(G, S), (mode, S)
        sizes[mode] = sol.sum(1).tolist()
    return f'sizes summed over the batch: {sizes}'


def test_3_greedy_follows_probabilities():
    G = nx.star_graph(5)                               # center 0
    g = batch_of([G])
    x = torch.tensor([[0.2], [0.9], [0.8], [0.1], [0.1], [0.1]])
    _, sol = env().evaluate(g, x)
    # greedy takes 1 (covers 0, 1), then 2, then the centre 0 (0.2 beats the leaves' 0.1), which
    # covers the rest; the descent then drops the redundant 1 and 2
    assert sol[:, 0].tolist() == [1, 0, 0, 0, 0, 0], sol
    x = torch.tensor([[0.05], [0.9], [0.8], [0.1], [0.2], [0.3]])
    _, sol = env().evaluate(g, x)
    assert sol[:, 0].tolist() == [0, 1, 1, 1, 1, 1], sol          # centre never chosen: all leaves
    x = torch.tensor([[0.95], [0.9], [0.8], [0.1], [0.1], [0.1]])
    _, sol = env().evaluate(g, x)
    assert sol[:, 0].tolist() == [1, 0, 0, 0, 0, 0], sol
    return 'ok'


def test_4_rlsa_finds_minimum():
    torch.manual_seed(0)
    out = []
    for seed in range(3):
        G = nx.barabasi_albert_graph(16, 2, seed=seed)
        gamma = min(len(S) for r in range(1, 17) for S in itertools.combinations(G, r) if is_dominating(G, S))
        g = batch_of([G])
        e = env('discrete')
        x = torch.randint(0, 2, (16, 16)).float()
        for t in range(300):
            tau = 1.0*(1 - t/300) + 1e-3
            _, grad = e.energy_func(g, x, True)
            delta = grad*(2*x - 1)/2
            term2 = -torch.kthvalue(-delta, 2, dim=0, keepdim=True).values
            x = torch.where(torch.rand_like(x) < torch.sigmoid((delta - term2)/tau), 1 - x, x)
        obj, _ = e.evaluate(g, x)
        best = int(round(obj.min().item()))
        assert best == gamma, (seed, best, gamma)
        out.append(f'{best}={gamma}')
    return 'RLSA best = domination number: ' + ', '.join(out)


def test_5_real_ba_graphs():
    root = os.environ.get('MDS_TEST_BA')
    if not root:
        return 'skipped (set MDS_TEST_BA)'
    from datasets.mds import Dataset
    ds = Dataset(root, torch.float32, num_samples=3)
    out = []
    for i in range(len(ds)):
        d = ds[i]
        G = nx.Graph()
        G.add_nodes_from(range(d.num_nodes))
        G.add_edges_from(d.edge_index.T.tolist())
        g = Batch.from_data_list([d])
        x = torch.randint(0, 2, (d.num_nodes, 3)).float()
        E, _ = env().energy_func(g, x, False)
        assert abs(E[0, 0].item() - brute_energy(G, x[:, 0].tolist())) < 1e-3
        obj, sol = env().evaluate(g, torch.rand(d.num_nodes, 4))
        S = [j for j in range(d.num_nodes) if sol[j, 0] > 0.5]
        assert is_dominating(G, S) and is_minimal(G, S)
        out.append(f'n={d.num_nodes} m={G.number_of_edges()} greedy DS {len(S)}')
    return '; '.join(out)


if __name__ == '__main__':
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            print(f'PASS {name} ({time.time()-t0:.1f}s): {fn()}', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name} ({time.time()-t0:.1f}s)', flush=True)
            traceback.print_exc()
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
