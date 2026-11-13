"""Idealised CAMPrior training on small graphs: exact label expectations, no sampling noise
(local addition, see cam/UPSTREAM.md; docs/label_correction.md section 7).

State space: all 2^n states of an n-node ER graph (n <= 12). The policy is a table of flip logits
L_k(x) (n per state and step); the reference a_k(x) is either state-independent (rate d/n) or
RLSA (utils/exact_small.flip_logits). One round of training:
  1. under the current policy, compute exactly E[label | X_k = x] for every state and step;
  2. move L towards the minimiser of the CAMPrior loss, a_k(x) + E[label]/tau (damped).
Labels (benefit of flipping node i at step k, conditioned on X_k = x):
  cam      (2x_i - 1) * E[grad_i(X_T)]                         CAM's lean label (with its sign factor)
  nosign   E[(1 - 2 f_i) (g(X_T) - g(X_T with i toggled))], f = flips of step k: the same without
           CAM's (-1)^{X_t xor X_1}; equals exact for a state-independent reference (CRN)
  exact    E[(1 - 2 f_i) (m_{k+1}(X') - m_{k+1}(X' with i toggled))], f = flips of step k,
           m = expected terminal energy under the current policy (what simulated labels estimate)
  dense    (2x_i - 1) * Rbar_k,  Rbar_k = E[(1 - lam) grad(X_{k+1}) + lam Rbar_{k+1}], Rbar_T = grad(X_T)
  target   (2x_i - 1) * E[sign(grad_i(X_T))]   unit-weight "move towards the terminal's best response"
Metrics of a policy: expected terminal energy, expected size after a greedy MIS decode, probability
that the decoded set is maximum, per-step KL to the reference.

    CUDA_VISIBLE_DEVICES=2 $CAM_PYTHON cam/scripts/mis/label_tabular.py --out runs/label_tabular/main
"""
import argparse
import itertools
import json
import os
import sys
import time

CAM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CAM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils import exact_small as ex  # noqa: E402


def er_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).double(), 1)
    A = up + up.T
    return A, -torch.ones(n, dtype=torch.float64)


def decode_sizes(A, X):
    """greedy MIS decode of every state: drop the node with most conflicts until independent, then
    add free nodes in index order; returns the set sizes"""
    An = A.cpu().numpy().astype(bool)
    n = An.shape[0]
    out = np.zeros(X.shape[0])
    for s, row in enumerate(X.cpu().numpy().astype(bool)):
        x = row.copy()
        while True:
            conf = (An[x][:, x]).sum(1)
            if conf.max(initial=0) == 0:
                break
            idx = np.flatnonzero(x)
            x[idx[conf.argmax()]] = False
        for i in range(n):
            if not x[i] and not (An[i] & x).any():
                x[i] = True
        out[s] = x.sum()
    return torch.tensor(out)


class Tabular:
    def __init__(self, A, b, beta, T, dev):
        self.n = A.shape[0]
        self.A, self.b, self.beta, self.T, self.dev = A.to(dev), b.to(dev), beta, T, dev
        self.X = ex.all_states(self.n).to(dev)                              # S x n
        S = 2**self.n
        idx = torch.arange(S, device=dev)
        self.xor = idx.unsqueeze(1) ^ idx.unsqueeze(0)                     # [x, x'] -> flip mask index
        self.toggle = idx.unsqueeze(1) ^ (1 << torch.arange(self.n, device=dev)).unsqueeze(0)   # S x n
        self.g = ex.energies(self.A, self.b, beta, self.X)                 # S
        self.grad = 2*beta*(self.X@self.A.T) + self.b.unsqueeze(0)          # S x n
        self.sgn = 2*self.X - 1

    def trans(self, L):
        """transition matrix P[x, x'] for flip logits L (S x n)"""
        lp = F.logsigmoid(-L).sum(1, keepdim=True) + (F.logsigmoid(L) - F.logsigmoid(-L))@self.X.T   # [x, f]
        return lp.exp().gather(1, self.xor)

    def kernels(self, Ls):
        return [self.trans(L) for L in Ls]

    def labels(self, P, kind, lam=1.0):
        """E[label | X_k = x] for k = 0..T-1, each S x n"""
        T = self.T
        if kind in ('cam', 'target'):
            h = self.grad if kind == 'cam' else torch.sign(self.grad)
            out = [None]*T
            for k in range(T - 1, -1, -1):
                h = P[k]@h                                                  # E[h(X_T) | X_k]
                out[k] = self.sgn*h
            return out
        if kind == 'nosign':
            H = self.g.unsqueeze(1) - self.g[self.toggle]                   # g(x) - g(x with i toggled), at X_T
            out = [None]*T
            for k in range(T - 1, -1, -1):
                out[k] = self.sgn*(P[k]@(self.sgn*H))                       # E[(1 - 2 f_i) H_i(X_{k+1})]
                H = P[k]@H                                                  # E[h(X_T) | X_k]
            return out
        if kind == 'dense':
            out = [None]*T
            R = P[T - 1]@self.grad                                          # E[grad(X_T) | X_{T-1}]
            out[T - 1] = self.sgn*R
            for k in range(T - 2, -1, -1):
                R = P[k]@((1 - lam)*self.grad + lam*R)
                out[k] = self.sgn*R
            return out
        if kind == 'exact':
            m = self.g
            out = [None]*T
            for k in range(T - 1, -1, -1):
                D = m.unsqueeze(1) - m[self.toggle]                         # at x': m(x') - m(x' toggled)
                # E over x' ~ P_k(.|x) of (1 - 2 f_i) D_i(x'), f_i = x_i xor x'_i, (1 - 2 f_i) = s_i s'_i
                out[k] = self.sgn*(P[k]@(self.sgn*D))
                m = P[k]@m
            return out
        raise ValueError(kind)

    def evaluate(self, P, dsize, alpha):
        mu = torch.full((2**self.n,), 1.0/2**self.n, device=self.dev, dtype=torch.float64)
        for k in range(self.T):
            mu = mu@P[k]
        return dict(energy=float(mu@self.g), size=float(mu@dsize), p_opt=float(mu[dsize >= alpha - 1e-9].sum()))


def kl_bern(L, a):
    p = torch.sigmoid(L)
    return (p*(F.logsigmoid(L) - F.logsigmoid(a)) + (1 - p)*(F.logsigmoid(-L) - F.logsigmoid(-a))).sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='runs/label_tabular/main')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--graphs', default='0.25:0,0.25:1,0.45:2')
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--d', type=int, default=2)
    ap.add_argument('--beta', type=float, default=1.02)
    ap.add_argument('--prior_tau0', type=float, default=1.0)
    ap.add_argument('--refs', default='static,rlsa')
    ap.add_argument('--labels', default='cam,nosign,exact,target,dense0.5,dense0.8')
    ap.add_argument('--taus', default='0.03,0.1,0.3,1,3')
    ap.add_argument('--rounds', type=int, default=40)
    ap.add_argument('--eta', type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(8)
    logf = open(os.path.join(a.out, 'log.txt'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'args {vars(a)} device {dev}')
    report = dict(args=vars(a), runs=[])
    for spec in a.graphs.split(','):
        p, gseed = float(spec.split(':')[0]), int(spec.split(':')[1])
        A, b = er_graph(a.n, p, gseed)
        tab = Tabular(A, b, a.beta, a.T, dev)
        dsize = decode_sizes(A, tab.X).to(dev)
        alpha = float(dsize.max())
        temps = [a.prior_tau0*(1 - k/a.T) for k in range(a.T)]
        for ref in a.refs.split(','):
            if ref == 'static':
                c = float(np.log(a.d/a.n) - np.log(1 - a.d/a.n))
                ref_L = [torch.full((2**a.n, a.n), c, device=dev, dtype=torch.float64) for _ in range(a.T)]
            else:
                ref_L = [ex.flip_logits(tab.A, tab.b, tab.X, temps[k], a.d, 0.0, a.beta) for k in range(a.T)]
            base = tab.evaluate(tab.kernels(ref_L), dsize, alpha)
            log(f'graph p={p} s={gseed} alpha={alpha:.0f} ref {ref}: reference policy {base}')
            for lab, tau in itertools.product(a.labels.split(','), [float(t) for t in a.taus.split(',')]):
                kind, lam = (('dense', float(lab[5:])) if lab.startswith('dense') else (lab, 1.0))
                L = [l.clone() for l in ref_L]
                hist = []
                t0 = time.time()
                for r in range(a.rounds):
                    P = tab.kernels(L)
                    if r % 10 == 0 or r == a.rounds - 1:
                        ev = tab.evaluate(P, dsize, alpha)
                        ev['round'] = r
                        ev['kl_per_step'] = float(np.mean([float(kl_bern(L[k], ref_L[k]).mean()) for k in range(a.T)]))
                        hist.append(ev)
                    lab_E = tab.labels(P, kind, lam)
                    L = [(1 - a.eta)*L[k] + a.eta*(ref_L[k] + lab_E[k]/tau) for k in range(a.T)]
                    if not all(torch.isfinite(l).all() for l in L):
                        log('  non-finite logits, stop')
                        break
                final = tab.evaluate(tab.kernels(L), dsize, alpha)
                # label check at the final policy: how far is each label from the exact one there
                Pf = tab.kernels(L)
                ex_f = torch.stack(tab.labels(Pf, 'exact'))
                cmp = {}
                for other in ('cam', 'nosign', 'target'):
                    o = torch.stack(tab.labels(Pf, other))
                    x, y = o.flatten().cpu().numpy(), ex_f.flatten().cpu().numpy()
                    cmp[other] = dict(corr=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan'),
                                      ratio_abs=float(np.abs(x).mean()/max(np.abs(y).mean(), 1e-12)))
                rr = dict(graph=f'p={p} seed={gseed}', alpha=alpha, ref=ref, label=lab, tau=tau, base=base,
                          final=final, history=hist, labels_vs_exact_at_final=cmp)
                report['runs'].append(rr)
                log(f'  {ref:6s} {lab:9s} tau {tau:5.2f}: size {final["size"]:.3f} p_opt {final["p_opt"]:.3f} '
                    f'energy {final["energy"]:.3f} (ref size {base["size"]:.3f}); at final: corr with exact '
                    f'cam {cmp["cam"]["corr"]:.2f} nosign {cmp["nosign"]["corr"]:.2f} target {cmp["target"]["corr"]:.2f} '
                    f'({time.time()-t0:.0f}s)')
            with open(os.path.join(a.out, 'report.json'), 'w') as f:
                json.dump(report, f, indent=1)
    log('done')


if __name__ == '__main__':
    main()
