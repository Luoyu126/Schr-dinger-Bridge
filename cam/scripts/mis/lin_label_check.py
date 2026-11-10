"""Small-graph check of the first-order labels with a nonzero residual (local addition, cam/UPSTREAM.md).

sim_label_check.py tests the labels at initialisation (residual 0), where the KL terms of the
first-order recursion vanish. Here the policy is p = sigmoid(a + r) with a fixed smooth residual
r_l(x, j) = c tanh((x W + u (1 + j/K))_l), so the exact cost-to-go under the policy,
    J_K = g,   J_j(x) = tau sum_l KL(p_l(x) || q_l(x)) + E_p[J_{j+1}(X_{j+1}) | X_j = x],
can be computed over all 2^n states, and the label J_{k+1}(x0) - J_{k+1}(x1) is exact. Compares the
first-order labels (own perturbation only, and with the neighbour-reaction term) and CAM's label
along trajectories of p. Writes <out>/report.json and <out>/report.md.

    $CAM_PYTHON cam/scripts/mis/lin_label_check.py --out runs/lin_label_check
"""
import argparse
import json
import os
import sys

CAM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CAM_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sim_label_check import er_graph, metrics  # noqa: E402
from solvers.lin_labels import kl_bern, linear_labels  # noqa: E402
from utils import exact_small as ex  # noqa: E402


def residual(X, W, u, c, j, K):
    return c*torch.tanh(X@W + u*(1 + j/K))


def exact_tables(A, b, temps, tau, d, beta, W, u, c):
    n, K = A.shape[0], len(temps)
    X = ex.all_states(n)
    idx = torch.arange(2**n)
    xor = idx.unsqueeze(1) ^ idx.unsqueeze(0)
    tabs = [None]*(K + 1)
    tabs[K] = ex.energies(A, b.reshape(-1), beta, X)
    for j in range(K - 1, -1, -1):
        a = ex.flip_logits(A, b, X, temps[j], d, 0.0, beta)
        z = a + residual(X, W, u, c, j, K)
        lp = F.logsigmoid(-z).sum(1, keepdim=True) + (F.logsigmoid(z) - F.logsigmoid(-z))@X.T
        tabs[j] = tau*kl_bern(z, a).sum(1) + (lp.exp()*tabs[j + 1][xor]).sum(1)
    return tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/lin_label_check')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--graphs', default='0.25:0,0.25:1,0.45:2')
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--chains', type=int, default=20)
    ap.add_argument('--d', type=int, default=2)
    ap.add_argument('--prior_tau0', type=float, default=1.0)
    ap.add_argument('--tau_train0', type=float, default=0.1)
    ap.add_argument('--settings', default='1.0:1.0,0.5:2.0,0.1:2.0', help='ratio:residual scale c')
    ap.add_argument('--beta', type=float, default=1.02)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(8)
    n, T, d, beta = args.n, args.T, args.d, args.beta
    inner = [args.prior_tau0*(1 - j/T) for j in range(T)]
    gen = torch.Generator().manual_seed(7)
    report = dict(args=vars(args), rows=[])
    for spec in args.graphs.split(','):
        p, gseed = float(spec.split(':')[0]), int(spec.split(':')[1])
        A, _, b = er_graph(n, p, gseed)
        W = torch.randn(n, n, generator=gen, dtype=torch.float64)/np.sqrt(n)
        u = torch.randn(n, generator=gen, dtype=torch.float64)
        for st in args.settings.split(','):
            ratio, c = [float(v) for v in st.split(':')]
            temps = [t*ratio for t in inner]
            tau = args.tau_train0*ratio
            tabs = exact_tables(A, b, temps, tau, d, beta, W, u, c)
            g = torch.Generator().manual_seed(100 + gseed)
            x = torch.randint(0, 2, (n, args.chains), generator=g).double()
            xs, zs, as_ = [x], [], []
            for j in range(T):
                a = ex.flip_logits(A, b, x.T, temps[j], d, 0.0, beta).T
                z = a + residual(x.T, W, u, c, j, T).T
                if j == T - 1:
                    grad_last = 2*beta*(A@x) + b
                as_.append(a)
                zs.append(z)
                x = torch.where(torch.rand(n, args.chains, generator=g, dtype=torch.float64) < torch.sigmoid(z), 1 - x, x)
                xs.append(x)
            X, Z, Aa = torch.stack(xs), torch.stack(zs), torch.stack(as_)
            C = T*args.chains
            x_next = X[1:].permute(1, 0, 2).reshape(n, C)
            x_now = X[:-1].permute(1, 0, 2).reshape(n, C)
            sel = np.tile(np.arange(n), (C, 1))
            sign = (1 - 2*(x_next != x_now).double()).T.numpy()
            starts = np.repeat(np.arange(1, T + 1), args.chains)
            exact = np.stack([ex.flip_label(tabs[s], x_next[:, q:q+1], sel[q:q+1], sign[q:q+1])[0].numpy()
                              for q, s in enumerate(starts)])
            labs = {'cam': (grad_last.unsqueeze(1)*(2*X[:-1].permute(1, 0, 2) - 1)).reshape(n, C).T.numpy()}
            for key, nb in (('lin', False), ('lin_nb', True)):
                labs[key] = (linear_labels(A, b, X, Z, Aa, temps, beta, beta, tau, neighbors=nb)
                             .permute(1, 0, 2).reshape(n, C).T.numpy())
            row = dict(p=p, seed=gseed, edges=int(A.sum().item()//2), ratio=ratio, c=c, tau_train=tau,
                       **{k: metrics(v, exact) for k, v in labs.items()})
            report['rows'].append(row)
            print(f'p={p} s={gseed} ratio={ratio} c={c}: corr cam {row["cam"]["corr"]:.3f} '
                  f'lin {row["lin"]["corr"]:.3f} lin_nb {row["lin_nb"]["corr"]:.3f}', flush=True)
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    lines = ['| graph | tau_train | c | label | corr | sign agree (nonzero) | MAE | mean abs exact |',
             '|---|---|---|---|---|---|---|---|']
    for r in report['rows']:
        for k, name in (('cam', 'CAM'), ('lin', 'first-order (own)'), ('lin_nb', 'first-order (+ neighbours)')):
            m = r[k]
            lines.append(f'| p={r["p"]} s={r["seed"]} ({r["edges"]} edges) | {r["tau_train"]:.3g} | {r["c"]:g} | '
                         f'{name} | {m["corr"]:.3f} | {m["sign_agree_nonzero"]:.3f} | {m["mae"]:.4g} | '
                         f'{m["ref_mean_abs"]:.3f} |')
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
