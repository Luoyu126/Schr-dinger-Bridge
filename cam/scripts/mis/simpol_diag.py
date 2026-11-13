"""Diagnostic of --label_mode simpol (local addition, see cam/UPSTREAM.md).

On a trained CAMPrior checkpoint, one training rollout per graph (as _generate_data_labels), then
for C sampled stored states and m sampled nodes each, the on-policy simulated label of
policy_cf_labels split into its two parts,
    label = sign * (dE + tau * dKL),   dE, dKL = factual minus toggled rollout,
repeated over R independent seeds of the rollout uniforms (common random numbers within a seed),
next to CAM's label grad_end * (2x - 1) on the same (state, node) pairs. Reports per graph: tau,
size of the rollout's total tau * KL, RMS of the two parts and of CAM's label, the seed noise of
the simulated label (RMS of its deviation from the seed mean) and how the seed mean agrees with
CAM's label (correlation, sign agreement).

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=6 $CAM_PYTHON cam/scripts/mis/simpol_diag.py \
        --checkpoint runs/cam/mis_er700/CAMPrior_p1_e100_s1/epoch_0049.pt --epoch 50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class _Parser(Exception):
    pass


def main_parser():
    """The argument parser built inside main.main(), captured at its parse_args call."""
    import main as M

    def grab(self, *a, **k):
        raise _Parser(self)
    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        M.main()
    except _Parser as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError('main.main() did not call parse_args')


@torch.no_grad()
def rollout(solver, graph):
    """The rollout of CAMPrior._generate_data_labels: states X_0..X_{T-1}, next states, and the
    energy gradient at X_{T-1} (where CAM's single-clip label takes it)."""
    n, Kc, T = graph.num_nodes, solver.num_kp, solver.num_tp
    ratio = solver.prior_tau_ratio()
    x = torch.randint(0, 2, size=(n, Kc), device=solver.accelerator.device, dtype=torch.float)
    states = []
    for t in range(T):
        states.append(x.clone())
        _, grad = solver.problem.energy_func(graph, x, True)
        logits, _ = solver.flip_logits(graph, x, torch.ones_like(x)*t/T, t, ratio,
                                       grad if solver.prior_problem is solver.problem else None)
        x = torch.where(torch.rand_like(x) < torch.sigmoid(logits), 1 - x, x)
    return states, states[1:] + [x], grad


@torch.no_grad()
def cost_parts(solver, graph, x_next, sel, starts, seeds, ratio):
    """policy_cf_labels with the two cost parts returned separately: energy and tau * KL, (C, m+1)."""
    from solvers.cam_prior import bernoulli_kl_logits
    from solvers.sim_labels import item_uniforms
    dev = x_next.device
    n, C = x_next.shape
    m, T, w = sel.shape[1], solver.num_tp, sel.shape[1] + 1
    starts = torch.as_tensor(starts, dtype=torch.long)
    order = torch.argsort(starts, stable=True)
    s_sorted = starts[order]
    U = item_uniforms(torch.as_tensor(seeds)[order], s_sorted, T, n).to(dev)
    X = x_next[:, order.to(dev)].float().repeat_interleave(w, dim=1)
    rows = sel[order.to(dev)].reshape(-1)
    cols = (torch.arange(C, device=dev).unsqueeze(1)*w + 1 + torch.arange(m, device=dev)).reshape(-1)
    X[rows, cols] = 1 - X[rows, cols]
    kl = torch.zeros(C*w, device=dev)
    for j in range(int(s_sorted.min()), T):
        P = int(torch.searchsorted(s_sorted, torch.tensor(j), right=True))
        if P == 0:
            continue
        for c0 in range(0, P*w, solver.simpol_chunk):
            c1 = min(c0 + solver.simpol_chunk, P*w)
            x = X[:, c0:c1]
            logits, prior = solver.flip_logits(graph, x, torch.ones_like(x)*j/T, j, ratio)
            ref = prior if solver.prior_kl_rlsa else torch.zeros_like(prior)
            kl[c0:c1] += float(solver.tau)*bernoulli_kl_logits(logits, ref).sum(0)
            u = U[:P, j, :].T.repeat_interleave(w, dim=1)[:, c0:c1]
            X[:, c0:c1] = torch.where(u < torch.sigmoid(logits), 1 - x, x)
    energy, _ = solver.problem.energy_func(graph, X, False)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(C)
    inv = inv.to(dev)
    return energy.reshape(C, w)[inv], kl.reshape(C, w)[inv]


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x*y).sum()/((x*x).sum()*(y*y).sum()).sqrt().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--epoch', type=int, required=True, help='training epoch whose tau_train is used')
    ap.add_argument('--graphs', type=int, default=2)
    ap.add_argument('--C', type=int, default=48, help='stored states per graph')
    ap.add_argument('--m', type=int, default=4, help='nodes per stored state')
    ap.add_argument('--R', type=int, default=6, help='independent seeds of the rollout uniforms')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='json with the per-graph statistics')
    a = ap.parse_args()
    ck = os.path.abspath(a.checkpoint)
    out = os.path.abspath(a.out) if a.out else None
    sys.path.insert(0, CAM_DIR)
    os.chdir(CAM_DIR)
    sys.argv = ['main.py']
    from solvers.cam_prior import CAMPrior
    data = os.environ.get('CAM_ER700_DIR', '/home/shengyuf/yunyi/data/CAM_data/data_er')
    save = os.path.join(CAM_DIR, '..', 'runs', 'diag', 'simpol_diag')
    argv = ['--train_path', f'{data}/er_train', '--valid_path', f'{data}/er_valid', '--test_path', f'{data}/er_test',
            '--problem', 'mis', '--samples_per_epoch', '1000', '--valid_sample', '4', '--batch_size', '1',
            '--mini_batch_size', '512', '--num_t', '50', '--num_k', '20', '--num_tp', '50', '--num_kp', '20',
            '--num_h', '64', '--num_l', '8', '--method', 'CAMPrior', '--model', 'EncodeProcessDecode',
            '--epochs', '100', '--N_anneal', '100', '--lr', '0.0005', '--prior_tau0', '1', '--seed', '1',
            '--save_dir', save, '--do_train', '--checkpoint', ck, '--label_mode', 'simpol',
            '--simpol_chunk', str(a.chunk), '--wandb_name', 'diag', '--wandb_run_name', 'simpol_diag']
    args = main_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    solver = CAMPrior(args)
    solver.model.to(solver.accelerator.device).eval()
    dev = solver.accelerator.device
    solver.tau = solver.temp_scheduler(a.epoch)
    ratio = solver.prior_tau_ratio()
    T, Kc = solver.num_tp, solver.num_kp
    print(f'checkpoint {ck}\nepoch {a.epoch}: tau_train {solver.tau:.4f}, prior ratio {ratio:.3f}', flush=True)
    rng = np.random.default_rng(a.seed)
    stats = []
    for gi, graph in enumerate(solver.train_loader):
        if gi >= a.graphs:
            break
        graph = graph.to(dev)
        n = graph.num_nodes
        graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
        states, nexts, grad_end = rollout(solver, graph)
        S = torch.stack(states, 1).reshape(n, T*Kc)
        Xn = torch.stack(nexts, 1).reshape(n, T*Kc)
        items = np.sort(rng.choice(T*Kc, a.C, replace=False))
        it = torch.as_tensor(items, device=dev)
        sel = torch.stack([torch.as_tensor(rng.choice(n, a.m, replace=False), device=dev) for _ in items])
        sign = 1 - 2*torch.gather((Xn[:, it] != S[:, it]).float(), 0, sel.T).T             # C x m
        starts = items//Kc + 1
        cam = torch.gather((grad_end[:, items % Kc]*(2*S[:, it] - 1)), 0, sel.T).T        # C x m
        dE, dK, totK, totE = [], [], [], []
        for r in range(a.R):
            seeds = 10**9*(r + 1) + 10**5*gi + items
            E, K = cost_parts(solver, graph, Xn[:, it], sel, starts, seeds, ratio)
            dE.append(sign*(E[:, :1] - E[:, 1:]))
            dK.append(sign*(K[:, :1] - K[:, 1:]))
            totK.append(K[:, 0])
            totE.append(E[:, 0])
        dE, dK = torch.stack(dE), torch.stack(dK)                                            # R x C x m
        lab = dE + dK
        mean = lab.mean(0)
        rms = lambda z: float(z.pow(2).mean().sqrt())
        st = dict(graph=gi, n=n, tau=float(solver.tau),
                  rollout_energy_mean=float(torch.stack(totE).mean()),
                  rollout_tauKL_mean=float(torch.stack(totK).mean()),
                  rollout_tauKL_start_le=[float(torch.stack(totK)[:, torch.as_tensor(starts <= s, device=dev)].mean())
                                          if (starts <= s).any() else None for s in (10, 25, 40)],
                  rms_dE=rms(dE), rms_tau_dKL=rms(dK), rms_label=rms(lab), rms_cam=rms(cam),
                  frac_dE_zero=float((dE == 0).float().mean()),
                  seed_noise_label=rms(lab - mean), seed_noise_dE=rms(dE - dE.mean(0)),
                  rms_seed_mean_label=rms(mean), rms_seed_mean_dE=rms(dE.mean(0)),
                  corr_mean_label_cam=corr(mean, cam), corr_mean_dE_cam=corr(dE.mean(0), cam),
                  sign_agree_mean_label_cam=float(((mean*cam) > 0).float()[cam != 0].mean()),
                  sign_agree_mean_dE_cam=float(((dE.mean(0)*cam) > 0).float()[cam != 0].mean()),
                  label_over_tau_rms=rms(lab)/float(solver.tau), cam_over_tau_rms=rms(cam)/float(solver.tau))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}), flush=True)
        stats.append(st)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(stats, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
