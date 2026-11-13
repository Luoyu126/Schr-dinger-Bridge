"""When is CAM's lean label right under a non-uniform reference? (local addition, cam/UPSTREAM.md;
plan in docs/label_correction.md section 6).

CAM's label for the flip decision of node i at step k is the "shortcut"
    cam_i(k)  = grad_i(X_T) * (2 x_i(k) - 1)
  = g(X_T with i set to x_i(k)) - g(X_T with i set to 1 - x_i(k))     (MIS energy is multilinear)
i.e. it assumes that the rest of the trajectory moves the same way whether or not i was flipped.
The quantity it stands for is the benefit of flipping under the reference process,
    exact_i(k) = m_{k+1}(X_{k+1} with i not flipped) - m_{k+1}(X_{k+1} with i flipped),
    m_j(x) = E_ref[g(X_T) | X_j = x].
Claim 1: if every node flips independently with a probability that does not depend on the state
(uniform, or any fixed per-node and per-step rates), two runs with common random numbers from the
two candidate next states stay exactly one node apart, so the parity-corrected shortcut
    par_i(k)  = grad_i(X_T) * (2 (x_i(k) XOR p_i) - 1),  p_i = X_T,i XOR X_{k+1},i
(p_i = 1 if i flips an odd number of times after step k) is an unbiased one-sample estimate of
exact_i(k). CAM's own label differs from it only when p_i = 1. Under RLSA (state-dependent) the
runs re-merge ("healing") and the shortcut is biased.
Claim 2: under RLSA the exact label is small next to the noise of one rollout, so a simulated
label needs many rollouts to carry signal.

  small   n = 12 ER graphs, every quantity exact by dynamic programming over all 2^n states:
          E1 bias of cam / par against exact, reference = static rates -> RLSA (mixing weight s);
          E2 healing: Hamming distance of CRN pairs after the perturbation;
          E3 correlation of K-rollout simulated labels with exact, K = 1 .. 256 (RLSA reference).
  large   ER-[700-800] test graphs, RLSA prior of the main runs (d = 20, tau0 = 1 linear): E2 healing
          and the shortcut against the CRN pair label, at several steps and two temperatures.

    $CAM_PYTHON cam/scripts/mis/d3_label_theory.py small --out runs/d3_label_theory/small
    CUDA_VISIBLE_DEVICES=2 $CAM_PYTHON cam/scripts/mis/d3_label_theory.py large --out runs/d3_label_theory/large
Writes <out>/report.json and <out>/log.txt; results in docs/label_correction.md section 6.
"""
import argparse
import glob
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


# ---------------------------------------------------------------- common
def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def label_stats(est, ref):
    est, ref = np.asarray(est, float).ravel(), np.asarray(ref, float).ravel()
    return dict(corr=corr(est, ref), mean_abs_est=float(np.abs(est).mean()),
                mean_abs_exact=float(np.abs(ref).mean()), mae=float(np.abs(est - ref).mean()),
                slope=float(np.dot(est, ref)/max(np.dot(ref, ref), 1e-12)))


class Logger:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(os.path.join(out, 'log.txt'), 'a')

    def __call__(self, msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()


# ---------------------------------------------------------------- small graphs (exact)
def static_logits(A, d, T):
    """state-independent per-node, per-step logits: rate ~ d/n scaled by (degree + 1), decaying in time"""
    n = A.shape[0]
    deg = A.sum(1) + 1
    w = deg/deg.mean()
    return [torch.logit(torch.clamp(d/n*w*(1 - 0.5*j/T), 1e-3, 0.5)) for j in range(T)]


def make_ref(kind, s, A, b, temps, d, beta, stat):
    """flip logits f(X, j) for rows X of states: 'uniform', or (1 - s) * static + s * RLSA"""
    n = A.shape[0]
    if kind == 'uniform':
        c = torch.logit(torch.tensor(d/n, dtype=torch.float64))
        return lambda X, j: torch.full_like(X, float(c))

    def f(X, j):
        z = stat[j].to(X).unsqueeze(0).expand_as(X)
        if s == 0:
            return z
        return (1 - s)*z + s*ex.flip_logits(A, b, X, temps[j], d, 0.0, beta)
    return f


def tables(A, b, beta, f, T, device):
    """m_j over all states, j = 0..T, for reference logits f"""
    n = A.shape[0]
    A, b = A.to(device), b.to(device)
    X = ex.all_states(n).to(device)
    idx = torch.arange(2**n, device=device)
    xor = idx.unsqueeze(1) ^ idx.unsqueeze(0)
    m = [None]*(T + 1)
    m[T] = ex.energies(A, b.reshape(-1), beta, X)
    for j in range(T - 1, -1, -1):
        z = f(X, j)
        lp = F.logsigmoid(-z).sum(1, keepdim=True) + (F.logsigmoid(z) - F.logsigmoid(-z))@X.T
        m[j] = (lp.exp()*m[j + 1][xor]).sum(1)
    return [t.cpu() for t in m]


def roll(f, x, j0, T, U):
    """reference run of the columns of x from step j0 to T with uniforms U[j] (n x C); all states"""
    xs = [x]
    for j in range(j0, T):
        z = f(x.T, j).T
        x = torch.where(U[j] < torch.sigmoid(z), 1 - x, x)
        xs.append(x)
    return xs


def grad_of(A, b, beta, x):
    return 2*beta*(A@x) + b


def small(args, log):
    n, T, d, beta = args.n, args.T, args.d, args.beta
    dev = torch.device(args.device)
    kinds = [('uniform', 0.0)] + [('mix', float(s)) for s in args.mix.split(',')]
    report = dict(args=vars(args), runs=[])
    for spec in args.graphs.split(','):
        p, gseed = float(spec.split(':')[0]), int(spec.split(':')[1])
        A, _, b = ex_graph(n, p, gseed)
        stat = static_logits(A, d, T)
        for ratio in [float(r) for r in args.ratios.split(',')]:
            temps = [args.prior_tau0*(1 - j/T)*ratio for j in range(T)]
            for kind, s in kinds:
                t0 = time.time()
                f = make_ref(kind, s, A, b, temps, d, beta, stat)
                m = tables(A, b, beta, (lambda X, j, f=f: f(X, j)), T, dev)
                gen = torch.Generator().manual_seed(1000*gseed + int(100*s) + (7 if kind == 'uniform' else 0))
                U = torch.rand(T, n, args.chains, generator=gen, dtype=torch.float64)
                x0 = torch.randint(0, 2, (n, args.chains), generator=gen).double()
                X = torch.stack(roll(f, x0, 0, T, U), 1)                       # n x T+1 x chains
                gT = grad_of(A, b, beta, X[:, T])                              # n x chains
                xk, xk1 = X[:, :-1], X[:, 1:]                                  # n x T x chains
                flipped = (xk != xk1).double()
                cam = gT.unsqueeze(1)*(2*xk - 1)
                par_bit = (X[:, T].unsqueeze(1) != xk1).double()
                par = gT.unsqueeze(1)*(2*((xk + par_bit) % 2) - 1)
                # exact label for every (i, k, chain): sign * (m(x_next) - m(x_next toggled i))
                exact = torch.zeros(n, T, args.chains, dtype=torch.float64)
                for k in range(T):
                    s_idx = ex.state_index(xk1[:, k])                          # chains
                    togg = s_idx.unsqueeze(0) ^ (1 << torch.arange(n)).unsqueeze(1)   # n x chains
                    exact[:, k] = (1 - 2*flipped[:, k])*(m[k + 1][s_idx].unsqueeze(0) - m[k + 1][togg])
                r = dict(graph=f'p={p} seed={gseed}', ratio=ratio, kind=kind, s=s,
                         odd_later_flips=float(par_bit.mean()),
                         one_rollout=dict(cam=label_stats(cam, exact), par=label_stats(par, exact)))

                # E1 bias and E2 healing on Q random items (k, chain, i), R CRN pairs each
                rng = np.random.default_rng(gseed*31 + int(100*s))
                Q, R = args.items, args.reps
                qk = rng.integers(0, T - 1, Q)
                qc = rng.integers(0, args.chains, Q)
                qi = rng.integers(0, n, Q)
                z_cam, z_par, b_cam, b_par, ex_q, hd_lag, merged = [], [], [], [], [], {}, []
                for k, c, i in zip(qk.tolist(), qc.tolist(), qi.tolist()):
                    xn = xk1[:, k, c]
                    xprev_i = xk[i, k, c].item()
                    x_nf = xn.clone()
                    x_nf[i] = xprev_i                                          # i not flipped at k
                    x_f = x_nf.clone()
                    x_f[i] = 1 - xprev_i                                       # i flipped at k
                    Ui = torch.rand(T, n, R, generator=gen, dtype=torch.float64)
                    pair = torch.cat([x_nf.unsqueeze(1).expand(n, R), x_f.unsqueeze(1).expand(n, R)], 1)
                    traj = roll(f, pair, k + 1, T, torch.cat([Ui, Ui], 2))
                    Z = traj[-1]
                    g_all = ex.energies(A, b.reshape(-1), beta, Z.T)
                    truth = g_all[:R] - g_all[R:]                               # CRN pair label
                    # shortcut along the factual run (whichever of the pair it is)
                    fact = slice(R, 2*R) if flipped[i, k, c] else slice(0, R)
                    Zf, xf_next = Z[:, fact], (x_f if flipped[i, k, c] else x_nf)
                    gi = grad_of(A, b, beta, Zf)[i]
                    pb = (Zf[i] != xf_next[i]).double()
                    lab_cam = gi*(2*xprev_i - 1)
                    lab_par = gi*(2*((xprev_i + pb) % 2) - 1)
                    e = exact[i, k, c].item()
                    for lab, zs, bs in ((lab_cam, z_cam, b_cam), (lab_par, z_par, b_par)):
                        mu, se = lab.mean().item(), lab.std().item()/np.sqrt(R)
                        zs.append((mu - e)/max(se, 1e-3))
                        bs.append(mu - e)
                    ex_q.append(e)
                    ham = torch.stack([(t[:, :R] != t[:, R:]).double().sum(0).mean() for t in traj]).numpy()
                    for lag in range(len(ham)):
                        hd_lag.setdefault(lag, []).append(ham[lag])
                    merged.append(float(((Z[:, :R] == Z[:, R:]).all(0)).double().mean()))
                    if kind == 'uniform' or s == 0:
                        assert np.allclose(ham, 1.0), ham                       # claim 1, pathwise
                        assert torch.allclose(truth, lab_par), 'CRN pair label != parity shortcut'
                ex_abs = float(np.mean(np.abs(ex_q)))
                r['bias'] = dict(items=Q, reps=R, mean_abs_exact=ex_abs,
                                 cam=dict(max_abs_z=float(np.max(np.abs(z_cam))),
                                          rel_bias=float(np.mean(np.abs(b_cam))/max(ex_abs, 1e-12))),
                                 par=dict(max_abs_z=float(np.max(np.abs(z_par))),
                                          rel_bias=float(np.mean(np.abs(b_par))/max(ex_abs, 1e-12))))
                r['healing'] = dict(hamming_by_lag={str(l): float(np.mean(v)) for l, v in sorted(hd_lag.items())
                                                    if l in (0, 1, 2, 3, 5, 10, 20)},
                                    merged_at_T=float(np.mean(merged)))
                log(f'{r["graph"]} ratio {ratio} {kind} s={s}: |exact| {ex_abs:.3f}; bias cam z {r["bias"]["cam"]["max_abs_z"]:.1f} '
                    f'rel {r["bias"]["cam"]["rel_bias"]:.2f}, par z {r["bias"]["par"]["max_abs_z"]:.1f} rel {r["bias"]["par"]["rel_bias"]:.2f}; '
                    f'corr(one) cam {r["one_rollout"]["cam"]["corr"]:.2f} par {r["one_rollout"]["par"]["corr"]:.2f}; '
                    f'hamming {r["healing"]["hamming_by_lag"]}; merged {r["healing"]["merged_at_T"]:.2f} '
                    f'({time.time()-t0:.0f}s)')

                # E3 (RLSA reference only): K-rollout simulated labels
                if kind == 'mix' and s == 1.0:
                    Q3, Kmax = args.e3_items, max(int(k) for k in args.e3_K.split(','))
                    qk = rng.integers(0, T - 1, Q3)
                    qc = rng.integers(0, args.chains, Q3)
                    qi = rng.integers(0, n, Q3)
                    sims_crn, sims_ind, ex3, cam3 = [], [], [], []
                    for k, c, i in zip(qk.tolist(), qc.tolist(), qi.tolist()):
                        xprev_i = xk[i, k, c].item()
                        x_nf = xk1[:, k, c].clone()
                        x_nf[i] = xprev_i
                        x_f = x_nf.clone()
                        x_f[i] = 1 - xprev_i
                        Ui = torch.rand(T, n, Kmax, generator=gen, dtype=torch.float64)
                        Uj = torch.rand(T, n, Kmax, generator=gen, dtype=torch.float64)
                        pair = torch.cat([x_nf.unsqueeze(1).expand(n, Kmax), x_f.unsqueeze(1).expand(n, Kmax)], 1)
                        Zc = roll(f, pair, k + 1, T, torch.cat([Ui, Ui], 2))[-1]
                        Zi = roll(f, pair, k + 1, T, torch.cat([Ui, Uj], 2))[-1]
                        gc = ex.energies(A, b.reshape(-1), beta, Zc.T)
                        gj = ex.energies(A, b.reshape(-1), beta, Zi.T)
                        sims_crn.append((gc[:Kmax] - gc[Kmax:]).numpy())
                        sims_ind.append((gj[:Kmax] - gj[Kmax:]).numpy())
                        ex3.append(exact[i, k, c].item())
                        cam3.append(cam[i, k, c].item())
                    sims_crn, sims_ind, ex3 = np.stack(sims_crn), np.stack(sims_ind), np.array(ex3)
                    r['e3'] = dict(items=Q3, mean_abs_exact=float(np.abs(ex3).mean()),
                                   one_rollout_noise_std_crn=float((sims_crn[:, 0] - ex3).std()),
                                   one_rollout_noise_std_ind=float((sims_ind[:, 0] - ex3).std()),
                                   cam_corr=corr(cam3, ex3),
                                   corr_crn={K: corr(sims_crn[:, :int(K)].mean(1), ex3) for K in args.e3_K.split(',')},
                                   corr_ind={K: corr(sims_ind[:, :int(K)].mean(1), ex3) for K in args.e3_K.split(',')})
                    log(f'   E3: |exact| {r["e3"]["mean_abs_exact"]:.3f}, one-rollout noise std crn '
                        f'{r["e3"]["one_rollout_noise_std_crn"]:.3f} ind {r["e3"]["one_rollout_noise_std_ind"]:.3f}; '
                        f'corr crn {r["e3"]["corr_crn"]}, ind {r["e3"]["corr_ind"]}, cam {r["e3"]["cam_corr"]:.2f}')
                report['runs'].append(r)
    return report


def ex_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).double(), 1)
    A = up + up.T
    return A, A.nonzero().T.contiguous(), -torch.ones(n, 1, dtype=torch.float64)


# ---------------------------------------------------------------- large graphs (CRN pairs)
def large(args, log):
    import networkx as nx
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = sorted(glob.glob(os.path.join(args.data, '*.gpickle')))[:args.num_graphs]
    if not files:
        raise SystemExit(f'no gpickles in {args.data}')
    T, d, beta = args.T, args.d, args.beta
    report = dict(args=vars(args), runs=[])
    for gi, fn in enumerate(files):
        import pickle
        with open(fn, 'rb') as fh:
            G = pickle.load(fh)
        G = nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
        A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float64, device=dev)
        b = -torch.ones(n, 1, device=dev, dtype=torch.float64)
        # envs/mis.py: edge weight 1/2 per directed edge, grad = 2 beta (A/2) x + b
        Ah = A/2

        def logits(x, j, temps):
            if args.ref == 'uniform':                         # sanity check of claim 1 at scale
                return torch.full_like(x, float(np.log(d/n) - np.log(1 - d/n)))
            grad = 2*beta*(Ah@x) + b
            delta = grad*(2*x - 1)/2
            thr = -torch.kthvalue(-delta, min(d, n), dim=0, keepdim=True).values
            return (delta - thr)/temps[j]

        def energy(x):
            return (x*(beta*(Ah@x) + b)).sum(0)

        for ratio in [float(r) for r in args.ratios.split(',')]:
            temps = [args.prior_tau0*(1 - j/T)*ratio for j in range(T)]
            gen = torch.Generator(device=dev).manual_seed(gi*100 + int(ratio*10))
            x = torch.randint(0, 2, (n, args.chains), generator=gen, device=dev).double()
            X = [x]
            for j in range(T):
                x = torch.where(torch.rand(n, args.chains, generator=gen, device=dev) < torch.sigmoid(logits(x, j, temps)), 1 - x, x)
                X.append(x)
            for k in [int(s) for s in args.steps.split(',')]:
                if k >= T - 1:
                    continue
                M, R = args.nodes, args.reps
                nodes = torch.stack([torch.randperm(n, generator=gen, device=dev)[:M] for _ in range(args.chains)], 1)  # M x chains
                xk, xk1 = X[k], X[k + 1]
                C = M*args.chains                                                    # items
                rows = nodes.reshape(-1).repeat_interleave(R)                        # item-major, R reps each
                cols = torch.arange(args.chains, device=dev).repeat(M).repeat_interleave(R)
                xprev = xk[rows, cols]
                flipped = xk1[rows, cols] != xprev
                base = xk1[:, cols].clone()                                          # n x C*R
                x_nf, x_f = base.clone(), base.clone()
                ar = torch.arange(C*R, device=dev)
                x_nf[rows, ar] = xprev
                x_f[rows, ar] = 1 - xprev
                pair = torch.cat([x_nf, x_f], 1)
                CR = C*R
                ham = [float((pair[:, :CR] != pair[:, CR:]).double().sum(0).mean())]
                for j in range(k + 1, T):
                    u = torch.rand(n, CR, generator=gen, device=dev)
                    pair = torch.where(torch.cat([u, u], 1) < torch.sigmoid(logits(pair, j, temps)), 1 - pair, pair)
                    ham.append(float((pair[:, :CR] != pair[:, CR:]).double().sum(0).mean()))
                e = energy(pair)
                truth = (e[:CR] - e[CR:]).reshape(C, R).cpu().numpy()                # CRN pair labels
                # shortcut on the factual run of each pair
                Z = torch.where(flipped.unsqueeze(0), pair[:, CR:], pair[:, :CR])
                x_fact = torch.where(flipped.unsqueeze(0), x_f, x_nf)
                gZ = (2*beta*(Ah@Z) + b)[rows, ar]
                pb = (Z[rows, ar] != x_fact[rows, ar]).double()
                cam = (gZ*(2*xprev - 1)).reshape(C, R).cpu().numpy()
                par = (gZ*(2*((xprev + pb) % 2) - 1)).reshape(C, R).cpu().numpy()
                merged = float((pair[:, :CR] == pair[:, CR:]).all(0).double().mean())
                if args.ref == 'uniform':
                    assert max(ham) == min(ham) == 1.0 and np.allclose(truth, par, atol=1e-6), \
                        f'claim 1 fails: hamming {min(ham)}..{max(ham)}, max|truth - par| {np.abs(truth - par).max()}'
                mt, se = truth.mean(1), truth.std(1, ddof=1)/np.sqrt(R)
                h = R//2
                ref_half = truth[:, h:].mean(1)
                r = dict(graph=os.path.basename(fn), n=n, ratio=ratio, k=k, items=C, reps=R,
                         energy_factual_T=float(energy(X[T]).mean()),
                         hamming_by_lag={str(l): ham[l] for l in (0, 1, 2, 3, 5, 10, 20, 40) if l < len(ham)},
                         hamming_at_T=ham[-1], merged_at_T=merged,
                         one_pair_abs=float(np.abs(truth).mean()), one_pair_noise_std=float(truth.std(1, ddof=1).mean()),
                         exact_rms_est=float(np.sqrt(max(np.mean(mt**2) - np.mean(se**2), 0.0))),
                         frac_items_signif=float(np.mean(np.abs(mt) > 2*se)),
                         cam_mean_abs=float(np.abs(cam).mean()),
                         corr_mean_cam_vs_mean_truth=corr(cam.mean(1), mt),
                         corr_one_cam_vs_mean_truth=corr(cam[:, 0], mt),
                         corr_mean_par_vs_mean_truth=corr(par.mean(1), mt),
                         sim_K_vs_heldout_half={str(K): corr(truth[:, :K].mean(1), ref_half)
                                                for K in (1, 2, 4, 8, 16, 32) if K <= h},
                         split_half_reliability=corr(truth[:, :h].mean(1), ref_half))
                report['runs'].append(r)
                log(f'{r["graph"]} n={n} ratio {ratio} k={k}: E(X_T) {r["energy_factual_T"]:.1f}, hamming {r["hamming_by_lag"]} merged {merged:.2f}; '
                    f'one pair |.| {r["one_pair_abs"]:.2f} noise {r["one_pair_noise_std"]:.2f}, E-label rms {r["exact_rms_est"]:.3f} '
                    f'(signif {r["frac_items_signif"]:.2f}); |cam| {r["cam_mean_abs"]:.2f}, corr(E cam, E truth) '
                    f'{r["corr_mean_cam_vs_mean_truth"]:.2f}, corr(one cam, E truth) {r["corr_one_cam_vs_mean_truth"]:.2f}; '
                    f'sim K vs held-out {r["sim_K_vs_heldout_half"]}')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('small')
    s.add_argument('--out', default='runs/d3_label_theory/small')
    s.add_argument('--device', default='cpu')
    s.add_argument('--n', type=int, default=12)
    s.add_argument('--graphs', default='0.25:0,0.25:1,0.45:2')
    s.add_argument('--T', type=int, default=50)
    s.add_argument('--d', type=int, default=2)
    s.add_argument('--beta', type=float, default=1.02)
    s.add_argument('--prior_tau0', type=float, default=1.0)
    s.add_argument('--ratios', default='1.0,0.1')
    s.add_argument('--mix', default='0,0.25,0.5,0.75,1')
    s.add_argument('--chains', type=int, default=20)
    s.add_argument('--items', type=int, default=40)
    s.add_argument('--reps', type=int, default=4000)
    s.add_argument('--e3_items', type=int, default=200)
    s.add_argument('--e3_K', default='1,4,16,64,256')
    l = sub.add_parser('large')
    l.add_argument('--out', default='runs/d3_label_theory/large')
    l.add_argument('--data', default=os.path.join(os.environ.get('CAM_ER700_DIR', ''), 'er_test'))
    l.add_argument('--num_graphs', type=int, default=4)
    l.add_argument('--T', type=int, default=50)
    l.add_argument('--d', type=int, default=20)
    l.add_argument('--beta', type=float, default=1.02)
    l.add_argument('--prior_tau0', type=float, default=1.0)
    l.add_argument('--ratios', default='1.0,0.1')
    l.add_argument('--steps', default='0,10,25,40,47')
    l.add_argument('--chains', type=int, default=10)
    l.add_argument('--nodes', type=int, default=16)
    l.add_argument('--reps', type=int, default=64, help='CRN pairs per item (E1/E3 at scale)')
    l.add_argument('--ref', default='rlsa', choices=['rlsa', 'uniform'])
    a = ap.parse_args()
    torch.set_num_threads(8)
    log = Logger(a.out)
    log(f'args {vars(a)}')
    report = small(a, log) if a.cmd == 'small' else large(a, log)
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    log('done')


if __name__ == '__main__':
    main()
