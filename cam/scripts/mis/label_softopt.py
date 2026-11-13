"""Which labels have the soft optimum of the CAMPrior objective as their fixed point? Exact check on
small graphs (local addition, see cam/UPSTREAM.md; docs/label_correction.md section 7).

CAMPrior minimises J(p) = E_p[g(X_T)] + tau * sum_k E_p[KL(p_k(.|X_k) || ref_k(.|X_k))], X_0 uniform.
Its minimiser over all (joint) transition kernels is the soft optimum
    p*_k(x'|x) = ref_k(x'|x) Z_{k+1}(x') / Z_k(x),  Z_T = exp(-g/tau),  Z_k = ref_k Z_{k+1},
    J* = E_{x0}[-tau log Z_0(x0)],
and CAM's adjoint is the soft value difference V*(x^(i)) - V*(x), V* = -tau log Z (CAM App. A.4).
A factorised policy matching the per-node marginals of p* is the target of the per-node loss; its
flip logit minus the reference logit, times tau, is the ideal label t*_k(x, i).

For every label kind of label_tabular.py (plus softexact: the label of soft policy iteration,
m = expected terminal energy plus tau * future KL under the current policy) this script reports
  at p*:   E_{p*}[label | X_k = x] against t* (visitation-weighted corr, slope, relative error),
           i.e. whether the soft optimum is (close to) a fixed point of the label;
  trained: the damped fixed-point iteration of label_tabular.py, and the gap J(policy) - J* of the
           policy it reaches, next to the gaps of the reference and of the marginal projection of p*.

    CUDA_VISIBLE_DEVICES= $CAM_PYTHON cam/scripts/mis/label_softopt.py --out runs/label_softopt/main
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from label_tabular import Tabular, decode_sizes, er_graph, kl_bern, ex  # noqa: E402


class Soft(Tabular):
    def logtrans(self, L):
        lp = F.logsigmoid(-L).sum(1, keepdim=True) + (F.logsigmoid(L) - F.logsigmoid(-L))@self.X.T
        return lp.gather(1, self.xor)

    def soft_opt(self, ref_L, tau):
        """joint soft-optimal kernels p*_k, J*, and the ideal per-node label t*_k (S x n)"""
        logZ = -self.g/tau
        P, t = [None]*self.T, [None]*self.T
        for k in range(self.T - 1, -1, -1):
            M = self.logtrans(ref_L[k]) + logZ.unsqueeze(0)
            logZk = torch.logsumexp(M, 1)
            P[k] = (M - logZk.unsqueeze(1)).exp()
            q1 = (P[k]@self.X).clamp(1e-300, 1 - 1e-16)                       # P(x'_i = 1 | x)
            flip = torch.where(self.X > 0.5, 1 - q1, q1).clamp(1e-300, 1 - 1e-16)
            t[k] = tau*(flip.log() - (1 - flip).log() - ref_L[k])
            logZ = logZk
        return P, float((-tau*logZ).mean()), t

    def labels_soft(self, P, kind, lam, L, ref_L, tau):
        if kind != 'softexact':
            return self.labels(P, kind, lam)
        m = self.g
        out = [None]*self.T
        for k in range(self.T - 1, -1, -1):
            D = m.unsqueeze(1) - m[self.toggle]
            out[k] = self.sgn*(P[k]@(self.sgn*D))
            m = P[k]@m + tau*kl_bern(L[k], ref_L[k])
        return out

    def visits(self, P):
        mu = torch.full((2**self.n,), 1.0/2**self.n, device=self.dev, dtype=torch.float64)
        out = []
        for k in range(self.T):
            out.append(mu)
            mu = mu@P[k]
        return out, mu

    def objective(self, L, ref_L, tau):
        P = self.kernels(L)
        mus, muT = self.visits(P)
        kl = sum(float(mus[k]@kl_bern(L[k], ref_L[k])) for k in range(self.T))
        return float(muT@self.g) + tau*kl, P


def wstats(lab, t, mus):
    """visitation-weighted corr, slope (lab on t) and relative error of lab against t"""
    w = torch.stack([m.unsqueeze(1).expand_as(t[0]) for m in mus]).flatten()
    x, y = torch.stack(lab).flatten(), torch.stack(t).flatten()
    w = w/w.sum()
    mx, my = (w*x).sum(), (w*y).sum()
    cxy, vx, vy = (w*(x - mx)*(y - my)).sum(), (w*(x - mx)**2).sum(), (w*(y - my)**2).sum()
    return dict(corr=float(cxy/(vx*vy).sqrt().clamp_min(1e-300)), slope=float(cxy/vy.clamp_min(1e-300)),
                rel_err=float(((w*(x - y)**2).sum()/(w*y**2).sum().clamp_min(1e-300)).sqrt()),
                sign_agree=float((w*((x*y) > 0).double()).sum()/(w*((x != 0) | (y != 0)).double()).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='runs/label_softopt/main')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--graphs', default='0.3:0,0.3:1,0.5:2')
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--d', type=int, default=2)
    ap.add_argument('--beta', type=float, default=1.02)
    ap.add_argument('--prior_tau0', type=float, default=1.0)
    ap.add_argument('--refs', default='static,rlsa')
    ap.add_argument('--labels', default='cam,nosign,dense0.5,target,exact,softexact')
    ap.add_argument('--taus', default='0.03,0.1,0.3,1,3')
    ap.add_argument('--rounds', type=int, default=60)
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
    labs = a.labels.split(',')
    for spec in a.graphs.split(','):
        p, gseed = float(spec.split(':')[0]), int(spec.split(':')[1])
        A, b = er_graph(a.n, p, gseed)
        tab = Soft(A, b, a.beta, a.T, dev)
        dsize = decode_sizes(A, tab.X).to(dev)
        alpha = float(dsize.max())
        temps = [a.prior_tau0*(1 - k/a.T) for k in range(a.T)]
        for ref, tau in itertools.product(a.refs.split(','), [float(x) for x in a.taus.split(',')]):
            if ref == 'static':
                c = float(np.log(a.d/a.n) - np.log(1 - a.d/a.n))
                ref_L = [torch.full((2**a.n, a.n), c, device=dev, dtype=torch.float64) for _ in range(a.T)]
            else:
                ref_L = [ex.flip_logits(tab.A, tab.b, tab.X, temps[k], a.d, 0.0, a.beta) for k in range(a.T)]
            Pstar, Jstar, tstar = tab.soft_opt(ref_L, tau)
            mus, muT = tab.visits(Pstar)
            ev_star = tab.evaluate(Pstar, dsize, alpha)
            proj_L = [r + ts/tau for r, ts in zip(ref_L, tstar)]
            J_proj, P_proj = tab.objective(proj_L, ref_L, tau)
            J_ref, P_ref = tab.objective(ref_L, ref_L, tau)
            at_star = {}
            for lab in labs:
                kind, lam = (('dense', float(lab[5:])) if lab.startswith('dense') else (lab, 1.0))
                # softexact at p* needs the KL of p*'s marginals; use the projection's logits
                E = tab.labels_soft(Pstar, kind, lam, proj_L, ref_L, tau)
                at_star[lab] = wstats(E, tstar, mus)
            rr = dict(graph=f'p={p} seed={gseed}', alpha=alpha, ref=ref, tau=tau, J_star=Jstar,
                      size_star=ev_star['size'], gap_ref=J_ref - Jstar, gap_proj=J_proj - Jstar,
                      size_ref=tab.evaluate(P_ref, dsize, alpha)['size'],
                      size_proj=tab.evaluate(P_proj, dsize, alpha)['size'], at_star=at_star, trained={})
            log(f'graph p={p} s={gseed} alpha={alpha:.0f} ref {ref} tau {tau}: J* {Jstar:.3f} size* {ev_star["size"]:.3f}; '
                f'gap ref {rr["gap_ref"]:.3f} proj {rr["gap_proj"]:.3f} (size {rr["size_proj"]:.3f}); at p*: ' +
                ' '.join(f'{l} c{v["corr"]:.2f}/s{v["slope"]:.2f}' for l, v in at_star.items()))
            for lab in labs:
                kind, lam = (('dense', float(lab[5:])) if lab.startswith('dense') else (lab, 1.0))
                L = [l.clone() for l in ref_L]
                ok = True
                for r in range(a.rounds):
                    P = tab.kernels(L)
                    E = tab.labels_soft(P, kind, lam, L, ref_L, tau)
                    L = [(1 - a.eta)*L[k] + a.eta*(ref_L[k] + E[k]/tau) for k in range(a.T)]
                    if not all(torch.isfinite(l).all() for l in L):
                        ok = False
                        break
                if not ok:
                    rr['trained'][lab] = dict(diverged=True)
                    log(f'    {lab:9s}: diverged')
                    continue
                J, P = tab.objective(L, ref_L, tau)
                ev = tab.evaluate(P, dsize, alpha)
                rr['trained'][lab] = dict(gap=J - Jstar, size=ev['size'], p_opt=ev['p_opt'],
                                         dist_to_proj=wstats([l - r for l, r in zip(L, ref_L)],
                                                             [ts/tau for ts in tstar], mus))
                log(f'    {lab:9s}: gap {J - Jstar:8.3f} size {ev["size"]:.3f} p_opt {ev["p_opt"]:.3f}')
            report['runs'].append(rr)
            with open(os.path.join(a.out, 'report.json'), 'w') as f:
                json.dump(report, f, indent=1)
    log('done')


if __name__ == '__main__':
    main()
