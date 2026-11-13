"""Wall-clock of CAM's TSP inference vs the local-search prior, on the same GPU and instances
(local addition, see cam/UPSTREAM.md).

CAM's inference (solvers/meta_solver.py ULSolver.evaluate_single, edge case; CAM paper App. B:
10 steps, 16 samples, then 1000 iterations of 2-opt) is timed with an untrained AnGNN of the paper's
size (12 layers, hidden 256). The network part costs the same for any weights. The decode part does
not: 2-opt stops early from good tours. It is reported two ways:
  cam_net        the 10 network forwards (exact)
  cam_merge      greedy merge of the 16 soft heatmaps (exact up to the heatmap)
  cam_2opt_rand  1000-iteration 2-opt from the untrained network's tours (upper bound: poor tours)
  cam_2opt_good  1000-iteration 2-opt from tours of the prior's quality before its final 2-opt
                 (proxy for a trained network, whose tours are already good)
The prior (scripts/tsp/prior_gate.py) is timed per component: merge, per-step 2-opt, final 2-opt.

    CUDA_VISIBLE_DEVICES=4 $CAM_PYTHON cam/scripts/tsp/time_compare.py --data .../tsp500_test_concorde.txt --n_inst 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from envs.tsp import batched_two_opt_torch, merge_tours  # noqa: E402
from models.angnn import AnGNN  # noqa: E402
from prior_gate import Instance, fast_two_opt, load, tour_len  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--n_inst', type=int, default=8)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--final_2opt', type=int, default=1000)
    ap.add_argument('--step_2opt', type=int, default=50)
    ap.add_argument('--prior_K', type=int, default=None, help='chains of the prior (default: --K)')
    ap.add_argument('--fast_step', type=int, default=1, help='prior per-step 2-opt: 1 fast_two_opt, 0 upstream')
    ap.add_argument('--net_chunk', type=int, default=16, help='samples per network forward (memory)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = AnGNN(n_layers=12, hidden_dim=256, out_channels=1, problem='tsp').to(dev).eval()
    rng = np.random.default_rng(0)
    tm = Timer()
    for pts, _ in load(a.data, a.n_inst):
        inst = Instance(pts, 50)
        g = Batch.from_data_list([Data(x=torch.from_numpy(pts).float(), edge_index=torch.from_numpy(inst.edge_index))]).to(dev)
        # ---- CAM inference
        sync(); t0 = time.time()
        with torch.no_grad():
            sol = torch.randint(0, 2, size=(inst.E, a.K), device=dev, dtype=torch.float)
            for t in range(a.T):
                tt = torch.ones_like(sol)*t/a.T
                p = torch.cat([torch.sigmoid(model(g, sol[:, c:c + a.net_chunk].unsqueeze(-1),
                                                   tt[:, c:c + a.net_chunk]).squeeze(-1))
                               for c in range(0, a.K, a.net_chunk)], 1)
                sol = sol*(1 - p) + p*(1 - sol) if t == a.T - 1 else torch.where(torch.rand_like(sol) < p, 1 - sol, sol)
        sync(); tm.add('cam_net', time.time() - t0)
        t0 = time.time()
        heat = sol.T.cpu().numpy()
        tours, _ = merge_tours(heat.reshape(-1) + 1e-6, pts, inst.edge_index, sparse_graph=True, parallel_sampling=a.K)
        tm.add('cam_merge', time.time() - t0)
        sync(); t0 = time.time()
        batched_two_opt_torch(pts, np.array(tours, dtype=np.int64), max_iterations=a.final_2opt, device=dev)
        sync(); tm.add('cam_2opt_rand', time.time() - t0)
        # ---- prior (prior_gate.run_prior with lam 0 -> 1, p_toward 0.5, d0 100), timed per component
        KP = a.prior_K or a.K
        x = rng.integers(0, 2, size=(KP, inst.E)).astype(np.float64)
        for k in range(a.T):
            lam = k/max(a.T - 1, 1)
            t0 = time.time()
            tours, _ = merge_tours((1 - lam + lam*x).reshape(-1) + 1e-6, pts, inst.edge_index, sparse_graph=True, parallel_sampling=KP)
            tm.add('prior_merge', time.time() - t0)
            sync(); t0 = time.time()
            if a.fast_step:
                tours = fast_two_opt(pts, np.array(tours, dtype=np.int64), a.step_2opt, dev)
            else:
                tours, _ = batched_two_opt_torch(pts, np.array(tours, dtype=np.int64), max_iterations=a.step_2opt, device=dev)
            sync(); tm.add('prior_step2opt', time.time() - t0)
            t0 = time.time()
            y = inst.indicator(tours)
            rate = np.where(x != y, 0.5, 100*(1 - k/a.T)/inst.E)
            x = np.where(rng.random(x.shape) < rate, 1 - x, x)
            tm.add('prior_update', time.time() - t0)
        t0 = time.time()
        tours, _ = merge_tours(x.reshape(-1) + 1e-6, pts, inst.edge_index, sparse_graph=True, parallel_sampling=KP)
        tm.add('prior_merge', time.time() - t0)
        sync(); t0 = time.time()
        final, _ = batched_two_opt_torch(pts, np.array(tours, dtype=np.int64), max_iterations=a.final_2opt, device=dev)
        sync(); tm.add('prior_final2opt', time.time() - t0)
        # proxy for a trained CAM's 2-opt: start from the prior's tours one step before convergence
        sync(); t0 = time.time()
        batched_two_opt_torch(pts, np.array(tours, dtype=np.int64), max_iterations=a.final_2opt, device=dev)
        sync(); tm.add('cam_2opt_good', time.time() - t0)
        print(f'instance done: prior best {tour_len(pts, final).min():.3f}', flush=True)
    n = a.n_inst
    per = {k: v/n for k, v in tm.t.items()}
    cam_lo = per['cam_net'] + per['cam_merge'] + per['cam_2opt_good']
    cam_hi = per['cam_net'] + per['cam_merge'] + per['cam_2opt_rand']
    prior = sum(v for k, v in per.items() if k.startswith('prior_'))
    print('seconds per instance: ' + '  '.join(f'{k} {v:.2f}' for k, v in sorted(per.items())))
    print(f'CAM inference {cam_lo:.2f} (good tours) .. {cam_hi:.2f} (untrained tours);  prior {prior:.2f}')


if __name__ == '__main__':
    main()
