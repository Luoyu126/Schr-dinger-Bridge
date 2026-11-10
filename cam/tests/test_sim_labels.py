"""Tests for solvers/sim_labels.py and the label options of solvers/cam_prior.py. CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_sim_labels.py

(Own runner like test_cam_prior.py; the test_* functions are also pytest-compatible.)
"""
import os
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU only, before torch is imported
os.environ.setdefault('WANDB_MODE', 'disabled')
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)
sys.path.insert(0, os.path.join(CAM_ROOT, 'tests'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solvers.sim_labels import (SimLabelPool, compute_task, item_uniforms, rlsa_cf_labels,  # noqa: E402
                                dense_adjacency)
from utils import exact_small  # noqa: E402

torch.set_num_threads(2)


def er_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), 1)
    A = up + up.T
    return A, A.nonzero().T.contiguous(), -torch.ones(n, 1)


def random_items(n, K, C, m, seed, start_min=1):
    g = torch.Generator().manual_seed(seed)
    x_next = torch.randint(0, 2, (n, C), generator=g).float()
    sel = torch.stack([torch.randperm(n, generator=g)[:m] for _ in range(C)])
    sign = 1 - 2*torch.randint(0, 2, (C, m), generator=g).float()
    starts = torch.randint(start_min, K + 1, (C,), generator=g)
    seeds = torch.randint(0, 2**40, (C,), generator=g)
    return x_next, sel, sign, starts, seeds


def naive_labels(A, b, x_next, sel, sign, starts, seeds, temps, d, eps, beta_prior, beta):
    """Column-by-column re-implementation (no batching, no sorting) with the same uniforms."""
    n, C = x_next.shape
    K = len(temps)
    U = item_uniforms(seeds, starts, K, n)
    out = torch.zeros(sel.shape)

    def run(x, c):
        x = x.clone()
        for j in range(int(starts[c]), K):
            grad = 2*beta_prior*(A@x) + b.reshape(-1)
            delta = grad*(2*x - 1)/2
            thr = torch.sort(delta, descending=True).values[min(d, n) - 1]
            prob = torch.sigmoid((delta - thr + eps)/temps[j])
            x = torch.where(U[c, j] < prob, 1 - x, x)
        return (x*(beta*(A@x) + b.reshape(-1))).sum()

    for c in range(C):
        g0 = run(x_next[:, c], c)
        for r in range(sel.shape[1]):
            x1 = x_next[:, c].clone()
            x1[sel[c, r]] = 1 - x1[sel[c, r]]
            out[c, r] = sign[c, r]*(g0 - run(x1, c))
    return out


SETTINGS = dict(d=3, eps=0.0, beta_prior=1.02, beta=1.02)


def test_1_kernel_equals_naive():
    n, K, C, m = 9, 7, 6, 3
    A, ei, b = er_graph(n, 0.4, 1)
    temps = [0.5*(1 - j/K) + 0.05 for j in range(K)]
    items = random_items(n, K, C, m, 2)
    fast = rlsa_cf_labels(A, b, *items, temps, **SETTINGS)
    slow = naive_labels(A, b, *items, temps, **SETTINGS)
    assert torch.allclose(fast, slow, atol=1e-5), (fast, slow)


def test_2_last_step_is_exact_energy_difference():
    n, K, C, m = 8, 5, 4, 8
    A, ei, b = er_graph(n, 0.5, 3)
    x_next, sel, sign, _, seeds = random_items(n, K, C, m, 4)
    starts = torch.full((C,), K)
    lab = rlsa_cf_labels(A, b, x_next, sel, sign, starts, seeds, [1.0]*K, **SETTINGS)
    g = exact_small.energies(A.double(), b.double(), 1.02, x_next.T.double())
    for c in range(C):
        for r in range(m):
            x1 = x_next[:, c].clone()
            x1[sel[c, r]] = 1 - x1[sel[c, r]]
            g1 = exact_small.energies(A.double(), b.double(), 1.02, x1.unsqueeze(0).double())[0]
            assert abs(lab[c, r] - sign[c, r]*(g[c] - g1)) < 1e-5


def test_3_unbiased_against_exact_expectation():
    """Mean of many simulated labels == exact base expectation difference (DP over all states)."""
    n, K, m, R = 8, 6, 8, 4000
    A, ei, b = er_graph(n, 0.45, 5)
    temps = [0.6*(1 - j/K) + 0.1 for j in range(K)]
    tab = exact_small.backward_tables(A, b, temps, **SETTINGS)
    x_next, sel, sign, _, _ = random_items(n, K, 3, m, 6)
    zs = []
    for c, start in zip(range(3), (1, 3, 5)):
        xs = x_next[:, c:c+1].repeat(1, R)
        lab = rlsa_cf_labels(A, b, xs, sel[c:c+1].repeat(R, 1), sign[c:c+1].repeat(R, 1),
                             torch.full((R,), start), torch.arange(R) + 10**6*c, temps, **SETTINGS)
        exact = exact_small.flip_label(tab[start], x_next[:, c:c+1], sel[c:c+1], sign[c:c+1])[0]
        mean, se = lab.double().mean(0), lab.double().std(0)/R**0.5 + 1e-9
        zs.append(((mean - exact)/se).abs().max().item())
    assert max(zs) < 4.5, zs


def _task(n=10, K=8, C=40, m=4, seed=7):
    A, ei, b = er_graph(n, 0.4, seed)
    temps = [0.4*(1 - j/K) + 0.05 for j in range(K)]
    x_next, sel, sign, starts, seeds = random_items(n, K, C, m, seed + 1)
    common = dict(edge_index=ei.numpy(), edge_weight=None, n=n, b=b.numpy(), temps=temps,
                  graph_key=1, **SETTINGS)
    arrs = (x_next.to(torch.uint8).numpy(), sel.numpy(), sign.numpy(), starts.numpy(), seeds.numpy())
    return common, arrs


def test_4_pool_equals_in_process_and_survives_faults():
    common, arrs = _task()
    ref = compute_task(dict(common, x_next=arrs[0], sel=arrs[1], sign=arrs[2], starts=arrs[3],
                            seeds=arrs[4]), torch.device('cpu'))
    logs = []
    pool = SimLabelPool(['cpu', 'cpu', 'cpu'], fallback=None, timeout=20, nthreads=1,
                        log=logs.append, faults={0: {'raise': 1}, 1: {'exit': 1}, 2: {'hang': 1}})
    try:
        out = pool.compute(common, *arrs)
        assert np.allclose(out, ref, atol=1e-6)
        assert pool.stats['restarts'] >= 2, (pool.stats, logs)    # the exiting and the hung worker
        out2 = pool.compute(common, *arrs)                          # after the restarts
        assert np.allclose(out2, ref, atol=1e-6)
    finally:
        pool.close()


def test_5_pool_drops_dead_workers_and_falls_back():
    common, arrs = _task(seed=11)
    ref = compute_task(dict(common, x_next=arrs[0], sel=arrs[1], sign=arrs[2], starts=arrs[3],
                            seeds=arrs[4]), torch.device('cpu'))
    logs = []
    pool = SimLabelPool(['cpu'], fallback=torch.device('cpu'), timeout=20, nthreads=1,
                        max_restarts=0, log=logs.append, faults={0: {'exit': 99}})
    try:
        out = pool.compute(common, *arrs)
        assert np.allclose(out, ref, atol=1e-6)
        assert pool.stats['dropped'] == 1 and pool.stats['fallback'] >= 1, (pool.stats, logs)
    finally:
        pool.close()


def test_6_masked_loss():
    from solvers.cam_prior import cam_prior_node_loss
    g = torch.Generator().manual_seed(0)
    n, B = 12, 5
    logits, prior, target = (torch.randn(B, n, generator=g) for _ in range(3))
    weight = torch.rand(B, n, generator=g)
    gi = torch.zeros(n, dtype=torch.long)
    full = cam_prior_node_loss(logits.T, prior.T, target.T, weight.T, gi, 0.1, 0.0, 30).item()
    ones = cam_prior_node_loss(logits.T, prior.T, target.T, weight.T, gi, 0.1, 0.0, 30,
                               mask=torch.ones(n, B), scale=1.0).item()
    assert abs(full - ones) < 1e-6
    # expectation over uniform masks of size m, rescaled by n/m, equals the full loss
    m, acc, R = 4, 0.0, 3000
    for _ in range(R):
        mask = torch.zeros(n, B)
        for c in range(B):
            mask[torch.randperm(n, generator=g)[:m], c] = 1
        acc += cam_prior_node_loss(logits.T, prior.T, target.T, weight.T, gi, 0.1, 0.0, 30,
                                   mask=mask, scale=n/m).item()
    assert abs(acc/R - full) < 0.05*abs(full) + 0.02, (acc/R, full)


def _camprior(data_dir, **over):
    import test_cam_prior as tcp
    return tcp.build(__import__('solvers.cam_prior', fromlist=['CAMPrior']).CAMPrior, data_dir, **over)


def test_7_camprior_label_paths():
    import test_cam_prior as tcp
    with tcp.TmpDir() as tmp:
        data = tcp.write_dataset(os.path.join(tmp, 'g'), [14], 0.3, 0)
        graph = tcp.load_batch(data)
        # cam labels with subsampling: same rollout and targets as the original path
        ref_solver = _camprior(data)
        torch.manual_seed(5)
        ref = ref_solver.generate_data(graph)
        sub = _camprior(data, label_frac=0.5)
        sub.model.load_state_dict(ref_solver.model.state_dict())
        torch.manual_seed(5)
        out = sub.generate_data(graph)
        assert len(out) == len(ref)
        m = round(0.5*14)
        for r, o in zip(ref, out):
            for a, bb in zip(r, o[:5]):
                assert torch.equal(a, bb)
            assert o[5].sum().item() == m
        assert sub.label_scale == 14/m
        loss = sub.train_batch(graph, [torch.stack(c) for c in zip(*[out[i] for i in range(4)])])
        assert torch.isfinite(loss)
        # sim labels: run, finite, nonzero only on the mask, one training step
        sim = _camprior(data, label_mode='sim', label_frac=0.3, save_dir=os.path.join(tmp, 'ck'))
        torch.manual_seed(5)
        out = sim.generate_data(graph)
        tgt, mask = torch.stack([o[3] for o in out]), torch.stack([o[5] for o in out])
        assert torch.isfinite(tgt).all() and (tgt[mask == 0] == 0).all()
        assert (tgt[mask == 1] != 0).any()
        loss = sim.train_batch(graph, [torch.stack(c) for c in zip(*[out[i] for i in range(6)])])
        loss.backward()
        assert torch.isfinite(loss)


def test_8_camprior_item_mapping():
    """Items handed to the label code match the stored states, flips and buffer rows."""
    import test_cam_prior as tcp
    import solvers.cam_prior as cp
    with tcp.TmpDir() as tmp:
        data = tcp.write_dataset(os.path.join(tmp, 'g'), [11], 0.35, 1)
        graph = tcp.load_batch(data)
        sim = _camprior(data, label_mode='sim', label_frac=0.5, save_dir=os.path.join(tmp, 'ck'))
        seen = {}
        orig = cp.simulated_labels

        def spy(pool, device, common, x_next, sel, sign, starts, seeds):
            lab = orig(pool, device, common, x_next, sel, sign, starts, seeds)
            seen.update(x_next=x_next, sel=sel, sign=sign, starts=starts, lab=lab, common=common)
            return lab
        cp.simulated_labels = spy
        try:
            torch.manual_seed(3)
            out = sim.generate_data(graph)
        finally:
            cp.simulated_labels = orig
        T, Kc, n = sim.num_tp, sim.num_kp, 11
        assert list(seen['starts']) == [t + 1 for t in range(T) for _ in range(Kc)]
        for t in range(T):
            for c in range(Kc):
                q = t*Kc + c                                   # item index (t-major)
                x, _, _, tgt, _, mask = out[c*T + t]           # buffer row (chain-major, one clip)
                sel = seen['sel'][q]
                assert set(sel.tolist()) == set(mask.nonzero().reshape(-1).tolist())
                assert np.allclose(tgt[sel].numpy(), seen['lab'][q], atol=1e-6)
                if t < T - 1:
                    x_next = out[c*T + t + 1][0]
                    assert np.array_equal(seen['x_next'][:, q], x_next.numpy().astype(np.uint8))
                    flipped = (x_next != x)[sel].float().numpy()
                    assert np.array_equal(seen['sign'][q], 1 - 2*flipped)
        # last step: no rollout, exact energy difference of the two candidate end states
        A, b = graph.A.to_dense(), graph.b
        for c in range(Kc):
            q = (T - 1)*Kc + c
            xT = torch.as_tensor(seen['x_next'][:, q]).float()
            for r, i in enumerate(seen['sel'][q]):
                x1 = xT.clone()
                x1[i] = 1 - x1[i]
                g0 = exact_small.energies(A.double(), b.double(), sim.args.beta, xT.unsqueeze(0).double())[0]
                g1 = exact_small.energies(A.double(), b.double(), sim.args.beta, x1.unsqueeze(0).double())[0]
                assert abs(seen['lab'][q][r] - seen['sign'][q][r]*(g0 - g1)) < 1e-4


def naive_linear(A, b, X, z, a, temps, beta_prior, beta, tau, neighbors=False):
    """Loop version of lin_labels.linear_labels (docs/label_correction.md, 3.5)."""
    import math
    K, n, C = z.shape
    sig = lambda v: 1/(1 + math.exp(-v))

    def kl(zz, aa):
        p, q = sig(zz), sig(aa)
        return p*math.log(p/q) + (1 - p)*math.log((1 - p)/(1 - q))

    def dprob(zz, dz, f):
        p, p2 = sig(zz), sig(zz + dz)
        return max(0.0, 1 - p2/p) if f else max(0.0, 1 - (1 - p2)/(1 - p))

    out = torch.zeros(K, n, C, dtype=torch.float64)
    for c in range(C):
        xK = X[K][:, c]
        gK = 2*beta*(A@xK) + b.reshape(-1)
        alpha = [(1 - 2*xK[i].item())*gK[i].item() for i in range(n)]
        for j in range(K - 1, -1, -1):
            x = X[j][:, c]
            f = [(X[j + 1][i, c] != x[i]).item() for i in range(n)]
            for i in range(n):
                out[j, i, c] = -(1 - 2*f[i])*alpha[i]
            if j == 0:
                break
            grad = 2*beta_prior*(A@x) + b.reshape(-1)
            new = []
            for i in range(n):
                zi, ai = z[j][i, c].item(), a[j][i, c].item()
                d_own = -2*(grad[i].item()*(2*x[i].item() - 1)/2)/temps[j]
                v = (1 - dprob(zi, d_own, f[i]))*alpha[i] + tau*(kl(zi + d_own, ai + d_own) - kl(zi, ai))
                for l in range(n):
                    if A[l, i] != 0:
                        da = beta_prior*A[l, i].item()*(1 - 2*x[i].item())*(2*x[l].item() - 1)/temps[j]
                        zl, al = z[j][l, c].item(), a[j][l, c].item()
                        v += tau*(kl(zl + da, al + da) - kl(zl, al))
                        if neighbors:
                            v += dprob(zl, da, f[l])*alpha[l]
                new.append(v)
            alpha = new
    return out


def _lin_case(n=7, K=5, C=3, seed=0):
    A, ei, b = er_graph(n, 0.45, seed)
    A, b = A.double(), b.double()
    g = torch.Generator().manual_seed(seed + 1)
    X = torch.randint(0, 2, (K + 1, n, C), generator=g).double()
    a = torch.randn(K, n, C, generator=g, dtype=torch.float64)*2
    z = a + torch.randn(K, n, C, generator=g, dtype=torch.float64)
    temps = [0.7*(1 - j/K) + 0.1 for j in range(K)]
    return A, b, X, z, a, temps


def test_9_linear_labels_equal_naive_loops():
    from solvers.lin_labels import linear_labels
    A, b, X, z, a, temps = _lin_case()
    for nb in (False, True):
        fast = linear_labels(A, b, X, z, a, temps, 1.02, 1.02, 0.07, chunk=2, neighbors=nb)
        slow = naive_linear(A, b, X, z, a, temps, 1.02, 1.02, 0.07, neighbors=nb)
        assert torch.allclose(fast, slow, atol=1e-9), (nb, (fast - slow).abs().max())


def test_10_linear_last_step_exact_and_no_kl_at_zero_residual():
    from solvers.lin_labels import linear_labels
    A, b, X, z, a, temps = _lin_case(seed=3)
    K = z.shape[0]
    out = linear_labels(A, b, X, z, a, temps, 1.02, 1.02, 0.07)
    xK, xk = X[K], X[K - 1]
    s = 1 - 2*(xK != xk).double()
    exact = s*(-(1 - 2*xK)*(2*1.02*(A@xK) + b))
    assert torch.allclose(out[K - 1], exact, atol=1e-10)
    # zero residual: the KL term vanishes, so tau does not matter
    o1 = linear_labels(A, b, X, a, a, temps, 1.02, 1.02, 0.01)
    o2 = linear_labels(A, b, X, a, a, temps, 1.02, 1.02, 5.0)
    assert torch.allclose(o1, o2, atol=1e-10)


def test_12_linear_own_term_is_contracting():
    """Zero residual, own perturbation only: |label| at every step <= |terminal adjoint| of that node."""
    from solvers.lin_labels import linear_labels
    for seed in range(4):
        A, b, X, z, a, temps = _lin_case(n=9, K=30, C=4, seed=seed)
        out = linear_labels(A, b, X, a, a, temps, 1.02, 1.02, 0.07)
        bound = ((1 - 2*X[-1])*(2*1.02*(A@X[-1]) + b)).abs()
        assert (out.abs() <= bound.unsqueeze(0) + 1e-12).all()


def test_11_camprior_linear_path():
    import test_cam_prior as tcp
    with tcp.TmpDir() as tmp:
        data = tcp.write_dataset(os.path.join(tmp, 'g'), [13], 0.3, 2)
        graph = tcp.load_batch(data)
        lin = _camprior(data, label_mode='lin')
        torch.manual_seed(4)
        out = lin.generate_data(graph)
        ref = _camprior(data)
        ref.model.load_state_dict(lin.model.state_dict())
        torch.manual_seed(4)
        ref_out = ref.generate_data(graph)
        for o, r in zip(out, ref_out):                       # same rollout, all nodes in the loss
            assert torch.equal(o[0], r[0]) and o[5].sum().item() == 13
        tgt = torch.stack([o[3] for o in out])
        assert torch.isfinite(tgt).all()
        loss = lin.train_batch(graph, [torch.stack(c) for c in zip(*[out[i] for i in range(6)])])
        loss.backward()
        assert torch.isfinite(loss)


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time()-t0:.1f}s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name}\n{traceback.format_exc()}', flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
