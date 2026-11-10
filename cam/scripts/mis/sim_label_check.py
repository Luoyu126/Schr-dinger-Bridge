"""Small-graph check of the training labels against exact values (local addition, cam/UPSTREAM.md).

On ER graphs with n <= 12 nodes every quantity of the RLSA base process is exact (utils/exact_small.py).
For trajectories of the CAMPrior sampler at initialisation (residual 0 = the RLSA prior, prior
tau0 = 1 as in the main runs, num_tp = 50, linear inner schedule; d scaled to the graph size) it
  1. computes the simulated labels through SimLabelPool on the given GPUs, with a fault injected
     into two workers, and checks they equal the in-process labels (same seeds);
  2. checks that the mean of many simulated labels equals the exact base expectation
     m_{k+1}(x0) - m_{k+1}(x1) (z-scores);
  3. compares CAM's lean label, one simulated label, its expectation and the exact optimal label
     V_{k+1}(x0) - V_{k+1}(x1) (soft value at tau_train) at three points of training.
Writes <out>/report.json and <out>/report.md.

    LABEL_GPUS=0,1,2,3 $CAM_PYTHON cam/scripts/mis/sim_label_check.py --out runs/sim_label_check
"""
import argparse
import json
import os
import sys
import time

CAM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CAM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solvers.sim_labels import SimLabelPool, compute_task  # noqa: E402
from utils import exact_small as ex  # noqa: E402


def er_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).double(), 1)
    A = up + up.T
    return A, A.nonzero().T.contiguous(), -torch.ones(n, 1, dtype=torch.float64)


def rollout(A, b, temps, d, eps, beta_prior, chains, seed):
    """Prior-only CAMPrior rollout: states X_0..X_T (n x T+1 x chains) and the grad at X_{T-1}."""
    g = torch.Generator().manual_seed(seed)
    n, T = A.shape[0], len(temps)
    x = torch.randint(0, 2, (n, chains), generator=g).double()
    xs = [x.clone()]
    for j in range(T):
        z = ex.flip_logits(A, b, x.T, temps[j], d, eps, beta_prior).T
        if j == T - 1:
            grad_last = (2*beta_prior*(A@x) + b).clone()
        x = torch.where(torch.rand(n, chains, generator=g, dtype=torch.float64) < torch.sigmoid(z), 1 - x, x)
        xs.append(x.clone())
    return torch.stack(xs, 1), grad_last


def metrics(est, ref):
    est, ref = np.asarray(est, float).ravel(), np.asarray(ref, float).ravel()
    nz = np.abs(ref) > 1e-9
    on = nz & (np.abs(est) > 1e-9)
    return dict(sign_agree_nonzero=float(np.mean(np.sign(est[on]) == np.sign(ref[on]))) if on.any() else float('nan'),
                frac_zero=float(np.mean(np.abs(est) <= 1e-9)),
                corr=float(np.corrcoef(est, ref)[0, 1]) if est.std() > 0 and ref.std() > 0 else float('nan'),
                sign_agree=float(np.mean(np.sign(est[nz]) == np.sign(ref[nz]))) if nz.any() else float('nan'),
                mae=float(np.mean(np.abs(est - ref))), ref_mean_abs=float(np.mean(np.abs(ref))),
                slope=float(np.dot(est, ref)/max(np.dot(ref, ref), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/sim_label_check')
    ap.add_argument('--gpus', default=os.environ.get('LABEL_GPUS', ''))
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--graphs', default='0.25:0,0.25:1,0.45:2')
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--chains', type=int, default=20)
    ap.add_argument('--d', type=int, default=2)
    ap.add_argument('--prior_tau0', type=float, default=1.0)
    ap.add_argument('--tau_train0', type=float, default=0.1)
    ap.add_argument('--ratios', default='1.0,0.5,0.1')
    ap.add_argument('--beta', type=float, default=1.02)
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--z_items', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gpus = [g for g in args.gpus.split(',') if g]
    dp_dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logf = open(os.path.join(args.out, 'log.txt'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'args {vars(args)}; pool GPUs {gpus}; DP on {dp_dev}')
    pool = None
    if gpus:
        # worker 1 raises on its first task, worker 2 exits on its first task (fault tolerance check)
        faults = {1: {'raise': 1}, 2: {'exit': 1}} if len(gpus) >= 3 else {}
        pool = SimLabelPool(gpus, fallback=None, timeout=600, log=log, faults=faults)
    report = dict(args=vars(args), graphs=[])
    n, T, eps, beta = args.n, args.T, 0.0, args.beta
    inner = [args.prior_tau0*(1 - j/T) for j in range(T)]
    try:
        for spec in args.graphs.split(','):
            p, gseed = float(spec.split(':')[0]), int(spec.split(':')[1])
            A, ei, b = er_graph(n, p, gseed)
            gi = dict(p=p, seed=gseed, edges=int(A.sum().item()//2), ratios=[])
            for ratio in [float(r) for r in args.ratios.split(',')]:
                temps = [t*ratio for t in inner]
                tau = args.tau_train0*ratio
                t0 = time.time()
                m_tab = [t.cpu() for t in ex.backward_tables(A.to(dp_dev), b.to(dp_dev), temps, args.d, eps, beta, beta)]
                v_tab = [t.cpu() for t in ex.backward_tables(A.to(dp_dev), b.to(dp_dev), temps, args.d, eps, beta, beta, tau=tau)]
                log(f'graph p={p} seed={gseed} ratio={ratio}: DP tables in {time.time()-t0:.1f}s')
                X, grad_last = rollout(A, b, temps, args.d, eps, beta, args.chains, 100 + gseed)
                # items (t, c), t-major; all nodes
                C = T*args.chains
                x_next = X[:, 1:, :].reshape(n, C)
                x_now = X[:, :-1, :].reshape(n, C)
                sel = np.tile(np.arange(n), (C, 1))
                sign = (1 - 2*(x_next != x_now).double()).T.numpy()          # C x n (sel = all nodes)
                starts = np.repeat(np.arange(1, T + 1), args.chains)
                seeds = 10**9*gseed + np.arange(C)
                common = dict(edge_index=ei.numpy(), edge_weight=None, n=n, b=b.float().numpy(),
                              temps=temps, d=args.d, eps=eps, beta_prior=beta, beta=beta,
                              graph_key=(gseed, ratio))
                arrs = (x_next.to(torch.uint8).numpy(), sel, sign.astype(np.float32), starts, seeds)
                ref = compute_task(dict(common, x_next=arrs[0], sel=sel, sign=arrs[2], starts=starts,
                                        seeds=seeds), dp_dev)
                same = None
                if pool is not None:
                    t0 = time.time()
                    lab = pool.compute(common, *arrs)
                    same = bool(np.allclose(lab, ref, atol=1e-5))
                    log(f'  pool labels {lab.shape} in {time.time()-t0:.2f}s, equal to in-process: {same}; '
                        f'pool stats {pool.stats}')
                    if not same:
                        raise SystemExit('pool labels differ from in-process labels')
                else:
                    lab = ref
                # exact references for every item
                e_mean = np.stack([ex.flip_label(m_tab[s], x_next[:, q:q+1], sel[q:q+1], sign[q:q+1])[0].numpy()
                                   for q, s in enumerate(starts)])
                e_soft = np.stack([ex.flip_label(v_tab[s], x_next[:, q:q+1], sel[q:q+1], sign[q:q+1])[0].numpy()
                                   for q, s in enumerate(starts)])
                # CAM's lean label: grad at X_{T-1} times (2 x_t - 1), for the same (t, c, i)
                cam = (grad_last.unsqueeze(1)*(2*X[:, :-1, :] - 1)).reshape(n, C).T.numpy()
                # unbiasedness: R replicate seeds for z_items random items
                rng = np.random.default_rng(gseed)
                qs = rng.choice(C, size=min(args.z_items, C), replace=False)
                R = args.reps
                rep_items = np.repeat(qs, R)
                rep_arrs = (arrs[0][:, rep_items], sel[rep_items], arrs[2][rep_items], starts[rep_items],
                            5*10**11 + np.arange(len(rep_items)))
                rep = (pool.compute(common, *rep_arrs) if pool is not None else
                       compute_task(dict(common, x_next=rep_arrs[0], sel=rep_arrs[1], sign=rep_arrs[2],
                                         starts=rep_arrs[3], seeds=rep_arrs[4]), dp_dev)).reshape(len(qs), R, n)
                mean, se = rep.mean(1), rep.std(1)/np.sqrt(R)
                diff = mean - e_mean[qs]
                # se floor 1e-3 (labels are energy differences of order 1): items whose R replicates
                # are all equal have se = 0, where a float32/float64 gap would give an infinite z
                z = diff/np.maximum(se, 1e-3)
                zmax = float(np.abs(z).max())
                i_max = np.unravel_index(np.abs(z).argmax(), z.shape)
                log(f'  unbiasedness: max |z| {zmax:.2f} at mean {mean[i_max]:.5f} exact {e_mean[qs][i_max]:.5f} '
                    f'se {se[i_max]:.2e}; max |mean - exact| {np.abs(diff).max():.4f}')
                late = starts > T//2
                ri = dict(ratio=ratio, tau_train=tau, pool_equal=same, z_max=zmax, z_tests=int(z.size),
                          max_abs_mean_err=float(np.abs(diff).max()),
                          one_rollout_noise_std=float(np.std(lab - e_mean)),
                          vs_exact_soft=dict(cam=metrics(cam, e_soft), sim_one=metrics(lab, e_soft),
                                             sim_mean=metrics(e_mean, e_soft)),
                          vs_exact_soft_late_steps=dict(cam=metrics(cam[late], e_soft[late]),
                                                        sim_one=metrics(lab[late], e_soft[late]),
                                                        sim_mean=metrics(e_mean[late], e_soft[late])),
                          sim_one_vs_mean=metrics(lab, e_mean))
                log(f'  z_max {zmax:.2f} over {z.size} tests; corr with exact optimal label: '
                    f'CAM {ri["vs_exact_soft"]["cam"]["corr"]:.3f}, sim(one) {ri["vs_exact_soft"]["sim_one"]["corr"]:.3f}, '
                    f'sim(mean) {ri["vs_exact_soft"]["sim_mean"]["corr"]:.3f}')
                gi['ratios'].append(ri)
            report['graphs'].append(gi)
    finally:
        if pool is not None:
            report['pool_stats'] = dict(pool.stats)
            pool.close()
    zs = [r['z_max'] for g in report['graphs'] for r in g['ratios']]
    report['pass_unbiased'] = bool(max(zs) < 5.0)
    report['pass_pool'] = all(r['pool_equal'] in (True, None) for g in report['graphs'] for r in g['ratios'])
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    lines = ['| graph | tau_train | z_max | label | corr | sign agree (nonzero) | zero | MAE | mean abs exact |',
             '|---|---|---|---|---|---|---|---|---|']
    for g in report['graphs']:
        for r in g['ratios']:
            for k, name in (('cam', 'CAM'), ('sim_one', 'sim, one rollout'), ('sim_mean', 'sim, expectation')):
                mm = r['vs_exact_soft'][k]
                lines.append(f'| p={g["p"]} s={g["seed"]} ({g["edges"]} edges) | {r["tau_train"]:.3g} | '
                             f'{r["z_max"]:.2f} | {name} | {mm["corr"]:.3f} | {mm["sign_agree_nonzero"]:.3f} | '
                             f'{mm["frac_zero"]:.2f} | {mm["mae"]:.3f} | {mm["ref_mean_abs"]:.3f} |')
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log('\n'.join(lines))
    log(f'unbiased: {report["pass_unbiased"]} (max z {max(zs):.2f}); pool equal: {report["pass_pool"]}; '
        f'pool stats {report.get("pool_stats")}')


if __name__ == '__main__':
    main()
