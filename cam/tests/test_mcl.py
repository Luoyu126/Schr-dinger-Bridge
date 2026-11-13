"""Tests for envs/mcl.py (maximum clique = MIS on the implicit complement graph). CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_mcl.py

Optional: MCL_TEST_RB=<dir of RB gpickles> also checks a few real RB graphs.
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

from envs.mcl import Env as MClEnv  # noqa: E402
from envs.mis import Env as MISEnv  # noqa: E402

torch.set_num_threads(2)
ARGS = types.SimpleNamespace(beta=1.02, mis_decoding='continuous')


def to_data(G):
    """the fields datasets/mis.py builds (edge weight 1/2 per directed edge, b = -1)"""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    e = torch.tensor([[u, v] for u, v in G.edges() if u != v] + [[v, u] for u, v in G.edges() if u != v],
                     dtype=torch.long).T.reshape(2, -1)
    return Data(edge_index=e, edge_weight=torch.full((e.shape[1],), 0.5), x=torch.ones(n, 1),
                b=-torch.ones(n, 1), num_nodes=n)


def batch_of(graphs):
    g = Batch.from_data_list([to_data(G) for G in graphs])
    g.A = torch.sparse_coo_tensor(g.edge_index, g.edge_weight,
                                  torch.Size((g.num_nodes, g.num_nodes))).to_sparse_csr()
    return g


def is_clique(G, nodes):
    return all(G.has_edge(u, v) for u, v in itertools.combinations(nodes, 2))


def split(graph, sol_row):
    """per-graph lists of selected local node ids from one row (N,) of a batched solution"""
    out, ptr = [], graph.ptr.tolist()
    for i in range(len(ptr) - 1):
        out.append([j - ptr[i] for j in range(ptr[i], ptr[i + 1]) if sol_row[j] > 0.5])
    return out


def random_graphs(seed):
    return [nx.gnp_random_graph(n, p, seed=seed + i) for i, (n, p) in enumerate([(9, 0.6), (13, 0.7), (7, 0.3)])]


def test_1_energy_and_grad_equal_mis_on_complement():
    """E and grad of envs/mcl.py == envs/mis.py on the explicit complement, batched, binary and soft x"""
    graphs = random_graphs(0)
    g, gc = batch_of(graphs), batch_of([nx.complement(G) for G in graphs])
    mcl, mis = MClEnv(ARGS), MISEnv(ARGS)
    torch.manual_seed(0)
    err = 0.0
    for x in (torch.randint(0, 2, (g.num_nodes, 5)).float(), torch.rand(g.num_nodes, 5)):
        e1, g1 = mcl.energy_func(g, x, True)
        e2, g2 = mis.energy_func(gc, x, True)
        assert e1.shape == e2.shape == (3, 5)
        err = max(err, (e1 - e2).abs().max().item(), (g1 - g2).abs().max().item())
    assert err < 1e-5, err
    # single graph (batch all zero, g_sum's other branch)
    G = graphs[1]
    g, gc = batch_of([G]), batch_of([nx.complement(G)])
    x = torch.randint(0, 2, (g.num_nodes, 4)).float()
    e1, g1 = mcl.energy_func(g, x, True)
    e2, g2 = mis.energy_func(gc, x, True)
    assert torch.allclose(e1, e2) and torch.allclose(g1, g2)
    # direct formula of RLSA eq. 19: -sum x + beta * #(non-adjacent pairs inside the selection)
    xb = x[:, 0]
    sel = [i for i in range(G.number_of_nodes()) if xb[i] > 0.5]
    pen = sum(1 for u, v in itertools.combinations(sel, 2) if not G.has_edge(u, v))
    assert abs(e1[0, 0].item() - (-len(sel) + 1.02*pen)) < 1e-5
    return f'max |diff| {err:.1e}'


def test_2_decoders_return_maximal_cliques():
    graphs = random_graphs(10)
    g = batch_of(graphs)
    torch.manual_seed(1)
    sizes = {}
    for mode in ('continuous', 'discrete'):
        env = MClEnv(types.SimpleNamespace(beta=1.02, mis_decoding=mode))
        x = torch.rand(g.num_nodes, 6) if mode == 'continuous' else torch.randint(0, 2, (g.num_nodes, 6)).float()
        obj, sol = env.evaluate(g, x.clone())
        if mode == 'continuous':      # layouts as envs/mis.py: continuous N x K, discrete K x N
            sol = sol.T
        assert sol.shape == (6, g.num_nodes)
        for k in range(6):
            for G, nodes in zip(graphs, split(g, sol[k])):
                assert is_clique(G, nodes), (mode, nodes)
                # maximal: no outside node is adjacent to all of the clique
                assert not any(all(G.has_edge(v, u) for u in nodes) for v in G if v not in nodes), (mode, nodes)
        if mode == 'continuous':
            assert torch.equal(obj, -sol.sum(1))
        sizes[mode] = sol.sum(1).tolist()
    return f'clique sizes summed over the batch: {sizes}'


def test_3_greedy_follows_probabilities():
    """continuous decode adds nodes by decreasing probability among the still-compatible ones"""
    G = nx.complete_graph(5)
    G.remove_edge(0, 1)                         # max cliques: {0,2,3,4} and {1,2,3,4}
    g = batch_of([G])
    env = MClEnv(ARGS)
    x = torch.tensor([[0.9], [0.95], [0.1], [0.2], [0.3]])
    _, sol = env.evaluate(g, x)
    assert sol[:, 0].tolist() == [0, 1, 1, 1, 1], sol
    return 'ok'


def test_4_rlsa_finds_planted_clique():
    """the RLSA step of solvers/rlsa.py with this env finds a planted clique of size 8"""
    torch.manual_seed(0)
    G = nx.gnp_random_graph(40, 0.3, seed=3)
    G.add_edges_from(itertools.combinations(range(8), 2))
    g = batch_of([G])
    env = MClEnv(ARGS)
    x = torch.randint(0, 2, (g.num_nodes, 8)).float()
    for t in range(200):
        tau = 2.0*(1 - t/200) + 1e-3
        _, grad = env.energy_func(g, x, True)
        delta = grad*(2*x-1)/2
        term2 = -torch.kthvalue(-delta, 2, dim=0, keepdim=True).values
        p = torch.sigmoid((delta-term2)/tau)
        x = torch.where(torch.rand_like(x) < p, 1-x, x)
    obj, sol = env.evaluate(g, x)
    best = int(-obj.min().item())
    omega = max(len(c) for c in nx.find_cliques(G))
    assert best == omega == 8, (best, omega)
    return f'RLSA best {best}, clique number {omega}'


def test_5_real_rb_graphs():
    root = os.environ.get('MCL_TEST_RB')
    if not root:
        return 'skipped (set MCL_TEST_RB)'
    from datasets.mcl import Dataset
    ds = Dataset(root, torch.float32, num_samples=3)
    mcl, mis = MClEnv(ARGS), MISEnv(ARGS)
    out = []
    for i in range(len(ds)):
        d = ds[i]
        G = nx.Graph()
        G.add_nodes_from(range(d.num_nodes))
        G.add_edges_from(d.edge_index.T.tolist())
        g, gc = batch_of([G]), batch_of([nx.complement(G)])
        x = torch.randint(0, 2, (g.num_nodes, 4)).float()
        e1, g1 = mcl.energy_func(g, x, True)
        e2, g2 = mis.energy_func(gc, x, True)
        assert torch.allclose(e1, e2, atol=1e-3) and torch.allclose(g1, g2, atol=1e-4)
        _, sol = mcl.evaluate(g, torch.rand(g.num_nodes, 4))
        nodes = [j for j in range(g.num_nodes) if sol[j, 0] > 0.5]
        assert is_clique(G, nodes)
        out.append(f'n={d.num_nodes} m={G.number_of_edges()} greedy clique {len(nodes)}')
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
