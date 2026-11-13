"""Gate check for a local-search prior on TSP, no network (local addition, see cam/UPSTREAM.md).

On MIS the CAMPrior reference is RLSA: the energy's flip gradient evaluated at the current state x_t,
while CAM's label is the same operator evaluated at the terminal state. The TSP analogue tried here
replaces the flip gradient by CAM's own terminal heuristic, applied to the current state:

    y_k = edge indicator of  2opt_{I}( merge(h_k) ),  h_k = (1 - lam_k) + lam_k x_k
          (merge = DIFUSCO greedy edge insertion; lam_k goes linearly from lam0 to lam1 over the
           steps: lam = 1 uses the state alone as the heatmap, lam = 0 ignores it, i.e. greedy
           insertion by distance)
    x_{k+1}: every edge with x_k != y_k flips with prob p_toward; every other edge flips with prob
             d_k / |E|, d_k = d0 (1 - k/T)  (random exploration, annealed to 0)

after T steps x_T is decoded the way CAM decodes at test time (merge, then final_2opt iterations of
2-opt). Edges live on the directed kNN graph of the dataset (sparse_factor neighbours); y marks both
directions of a tour edge, so it does not depend on the direction the tour is read in.

Controls (same final decode):
  random  x_0 uniform, decoded directly (what an untrained CAM policy roughly produces)
  greedy  all-zero heatmap (greedy edge insertion by distance), 2-opt with the prior's total 2-opt
          budget (final_2opt + T * I); deterministic, one chain

Report per method: mean over instances of the best and of the mean over K chains of the tour length,
before and after the final 2-opt, and the gap to the reference (concorde) tour of the data file.

    CUDA_VISIBLE_DEVICES=4 $CAM_PYTHON cam/scripts/tsp/prior_gate.py --data .../tsp500_valid_concorde.txt \
        --n_inst 32 --step_2opt 50,200 --p_toward 0.5,1 --d0 100,1000 --lam 1:1,0:1 --out runs/tsp_gate/valid500
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.neighbors import KDTree

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, CAM_DIR)
from envs.tsp import batched_two_opt_torch, merge_tours  # noqa: E402


def fast_two_opt(points, tours, max_iterations, device):
    """Best-improvement 2-opt like envs/tsp.py batched_two_opt_torch (same move set and choice), but
    vectorised over the batch and applied only to tours that still have an improving move (the
    upstream loop reverses a segment in every tour whenever any tour of the batch improves)."""
    with torch.inference_mode():
        P = torch.as_tensor(points, device=device)
        T = torch.as_tensor(tours, device=device)
        B, n1 = T.shape
        n = n1 - 1
        pos = torch.arange(n1, device=device)
        for _ in range(max_iterations):
            a, b = P[T[:, :-1]], P[T[:, 1:]]                                  # B x n x 2
            d_ab = (a - b).norm(dim=-1)                                       # edge (i, i+1)
            change = (torch.cdist(a, a) + torch.cdist(b, b) - d_ab.unsqueeze(2) - d_ab.unsqueeze(1))
            change = torch.triu(change, diagonal=2)
            best, arg = change.reshape(B, -1).min(1)
            act = best < -1e-6
            if not bool(act.any()):
                break
            i, j = (arg // n).unsqueeze(1), (arg % n).unsqueeze(1)
            seg = (pos > i) & (pos <= j) & act.unsqueeze(1)
            T = torch.gather(T, 1, torch.where(seg, i + 1 + j - pos, pos))
        return T.cpu().numpy()


def load(path, n_inst):
    out = []
    for line in open(path).read().splitlines()[:n_inst]:
        pts, tour = line.strip().split(' output ')
        pts = np.array(pts.split(' '), dtype=np.float64).reshape(-1, 2)
        tour = np.array(tour.split(' '), dtype=np.int64) - 1
        out.append((pts, tour))
    return out


def tour_len(pts, tours):
    """tours: (B, n+1) closed tours"""
    d = pts[tours[:, 1:]] - pts[tours[:, :-1]]
    return np.sqrt((d**2).sum(-1)).sum(-1)


class Instance:
    def __init__(self, pts, sparse_factor):
        self.pts = pts
        self.n = len(pts)
        _, idx = KDTree(pts, leaf_size=30, metric='euclidean').query(pts, k=sparse_factor)
        self.e0 = np.repeat(np.arange(self.n), sparse_factor)
        self.e1 = idx.reshape(-1)
        self.edge_index = np.stack([self.e0, self.e1])
        self.E = len(self.e0)
        self.key = {(a, b): i for i, (a, b) in enumerate(zip(self.e0, self.e1))}
        self.hash = self.e0*self.n + self.e1

    def decode(self, x, n_2opt, device, fast=False):
        """x: (K, E) heatmaps -> closed tours (K, n+1) after merge and n_2opt 2-opt iterations
        (upstream batched_two_opt_torch, or fast_two_opt with fast=True)"""
        K = x.shape[0]
        tours, _ = merge_tours(x.reshape(-1) + 1e-6, self.pts, self.edge_index, sparse_graph=True, parallel_sampling=K)
        tours = np.array(tours, dtype=np.int64)
        if n_2opt > 0:
            if fast:
                tours = fast_two_opt(self.pts, tours, n_2opt, device)
            else:
                tours, _ = batched_two_opt_torch(self.pts, tours, max_iterations=n_2opt, device=device)
        return tours

    def indicator(self, tours):
        """(K, n+1) tours -> (K, E) 0/1 edge indicator, both directions of each tour edge"""
        a, b = tours[:, :-1], tours[:, 1:]
        h = np.concatenate([a*self.n + b, b*self.n + a], 1)
        return np.stack([np.isin(self.hash, hk) for hk in h]).astype(np.float64)


def run_prior(inst, K, T, n_step, p_toward, d0, lam0, lam1, rng, device, trace, fast=False):
    x = rng.integers(0, 2, size=(K, inst.E)).astype(np.float64)
    for k in range(T):
        lam = lam0 + (lam1 - lam0)*k/max(T - 1, 1)
        tours = inst.decode(1 - lam + lam*x, n_step, device, fast)
        trace.append(float(tour_len(inst.pts, tours).mean()))
        y = inst.indicator(tours)
        rate = np.where(x != y, p_toward, d0*(1 - k/T)/inst.E)
        x = np.where(rng.random(x.shape) < rate, 1 - x, x)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_inst', type=int, default=32)
    ap.add_argument('--sparse_factor', type=int, default=50)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--final_2opt', type=int, default=1000)
    ap.add_argument('--step_2opt', default='0,50')
    ap.add_argument('--p_toward', default='0.5,1')
    ap.add_argument('--d0', default='0,100,500')
    ap.add_argument('--lam', default='1:1,0:1', help='lam0:lam1 pairs')
    ap.add_argument('--controls', default='random,greedy')
    ap.add_argument('--fast_step', default='0', help='per-step 2-opt implementation(s): 0 upstream, 1 fast_two_opt')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logf = open(os.path.join(a.out, 'log.txt'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'args {vars(a)} device {device}')
    data = load(a.data, a.n_inst)
    insts = [Instance(p, a.sparse_factor) for p, _ in data]
    ref = np.array([tour_len(p, np.append(t, t[0])[None])[0] if t[0] != t[-1] else tour_len(p, t[None])[0]
                    for p, t in data])
    log(f'{len(data)} instances, n={insts[0].n}, |E|={insts[0].E}, reference length {ref.mean():.4f}')

    configs = [dict(method=c) for c in a.controls.split(',') if c]
    for s, p, d, lm, fs in itertools.product(a.step_2opt.split(','), a.p_toward.split(','), a.d0.split(','),
                                             a.lam.split(','), a.fast_step.split(',')):
        cfg = dict(method='prior', step_2opt=int(s), p_toward=float(p), d0=float(d),
                   lam0=float(lm.split(':')[0]), lam1=float(lm.split(':')[1]))
        if fs == '1':
            cfg['fast_step'] = True
        configs.append(cfg)
    path = os.path.join(a.out, 'report.json')
    report = json.load(open(path)) if os.path.exists(path) else dict(args=vars(a), ref=float(ref.mean()), results=[])
    done = {json.dumps(r['config'], sort_keys=True) for r in report['results']}

    for cfg in configs:
        if json.dumps(cfg, sort_keys=True) in done:
            continue
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        pre_best, pre_mean, post_best, post_mean, traces = [], [], [], [], []
        for inst in insts:
            trace = []
            if cfg['method'] == 'random':
                x, n_final = rng.integers(0, 2, size=(a.K, inst.E)).astype(np.float64), a.final_2opt
            elif cfg['method'] == 'greedy':
                budget = a.final_2opt + a.T*max(int(s) for s in a.step_2opt.split(','))
                x, n_final = np.zeros((1, inst.E)), budget
            else:
                x = run_prior(inst, a.K, a.T, cfg['step_2opt'], cfg['p_toward'], cfg['d0'],
                              cfg['lam0'], cfg['lam1'], rng, device, trace, cfg.get('fast_step', False))
                n_final = a.final_2opt
            pre = tour_len(inst.pts, inst.decode(x, 0, device))
            post = tour_len(inst.pts, inst.decode(x, n_final, device))
            pre_best.append(pre.min()); pre_mean.append(pre.mean())
            post_best.append(post.min()); post_mean.append(post.mean())
            traces.append(trace)
        gap = lambda v: float((np.array(v)/ref - 1).mean()*100)  # noqa: E731
        res = dict(config=cfg, seconds=time.time() - t0,
                   pre_best=float(np.mean(pre_best)), pre_mean=float(np.mean(pre_mean)),
                   post_best=float(np.mean(post_best)), post_mean=float(np.mean(post_mean)),
                   gap_pre_best=gap(pre_best), gap_post_best=gap(post_best), gap_post_mean=gap(post_mean),
                   trace=np.mean(traces, 0).tolist() if traces[0] else [])
        report['results'].append(res)
        with open(path, 'w') as f:
            json.dump(report, f, indent=1)
        log(f'{cfg}: pre best {res["pre_best"]:.3f} ({res["gap_pre_best"]:.2f}%)  post best {res["post_best"]:.3f} '
            f'({res["gap_post_best"]:.2f}%)  post mean {res["post_mean"]:.3f} ({res["gap_post_mean"]:.2f}%)  '
            f'trace {" ".join(f"{v:.2f}" for v in res["trace"])}  [{res["seconds"]:.0f}s]')
    log('done')


if __name__ == '__main__':
    main()
