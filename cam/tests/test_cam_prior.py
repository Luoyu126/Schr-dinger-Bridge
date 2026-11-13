"""Tests for solvers/cam_prior.py (CAM + RLSA energy prior). CPU only.

Run from the repository root with the CAM conda env (used read-only; pytest is not installed in
it, so this file has its own runner; the test_* functions are also pytest-compatible):

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_cam_prior.py

Temporary files go to $TMPDIR (env.sh points it under /usr0/home/shengyuf/yunyi/.cache/cam),
or to runs/scratch/cam_tests in the repository if TMPDIR is not set.
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU only, before torch is imported
os.environ.setdefault('WANDB_MODE', 'disabled')
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(CAM_ROOT)
sys.path.insert(0, CAM_ROOT)
if not os.environ.get('TMPDIR'):
    tempfile.tempdir = os.path.join(REPO_ROOT, 'runs', 'scratch', 'cam_tests')
    os.makedirs(tempfile.tempdir, exist_ok=True)

import networkx as nx  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from datasets.mis import Dataset  # noqa: E402
from solvers.cam import CAM  # noqa: E402
from solvers.cam_prior import CAMPrior, PRIOR_META_SUFFIX, add_cam_prior_args  # noqa: E402
from solvers.rlsa import RLSA  # noqa: E402
from utils.temp_schedulers import TempScheduler  # noqa: E402

torch.set_num_threads(2)


# ---------------------------------------------------------------- helpers
def write_dataset(root, sizes, p, seed):
    """Write ER graphs as CAM gpickles (same layout as scripts/export_cam_dataset.py)."""
    os.makedirs(root, exist_ok=True)
    for i, n in enumerate(sizes):
        G = nx.gnp_random_graph(n, p, seed=seed + i)
        nx.write_gpickle(G, os.path.join(root, f'g_{i:03d}.gpickle'))
    return root


def load_batch(root, idxs=None):
    ds = Dataset(root, torch.float32)
    idxs = range(len(ds)) if idxs is None else idxs
    graph = Batch.from_data_list([ds[i] for i in idxs])
    graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight,
                                      torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
    return graph


def make_args(data_dir, **over):
    """main.py defaults (subset used by the solvers) plus the CAMPrior flags' real defaults."""
    a = dict(seed=0, train_path=data_dir, train_samples=-1, valid_path=data_dir, valid_samples=-1,
             test_path=data_dir, test_samples=-1, problem='mis', num_workers=0,
             save_dir=os.path.join(data_dir, 'ckpt'), log_file=os.path.join(data_dir, 'log.txt'),
             wandb_name='camprior_test', wandb_run_name='camprior_test', wandb_entity=None,
             num_t=6, num_k=4, repetition=1, num_d=30, tau0=0.1, tau1=0.0, temp_schedule='linear',
             lambd=0.0, beta=1.02, inference_schedule='linear', skip_decode=False, sparse_factor=50,
             two_opt_iterations=100, mis_decoding='continuous', method='CAMPrior',
             model='EncodeProcessDecode', num_tp=6, num_kp=4, num_l=2, num_h=16, time_scale=1000,
             diffusion_schedule='linear', optimizer='AdamW', lr_scheduler='cosine-decay',
             grad_clip=None, data_precision='fp32', do_train=True, checkpoint=None, epochs=2,
             batch_size=1, mini_batch_size=8, lr=1e-3, weight_decay=1e-4, patience=20,
             gradient_accumulation_steps=1, use_activation_checkpoint=False, compile=False,
             samples_per_epoch=2, num_clips=1, N_warmup=0, N_anneal=2, N_equil=0,
             noise_energy='bernoulli', inner_loop_steps=1, batches_per_step=None)
    a.update(vars(add_cam_prior_args(argparse.ArgumentParser()).parse_args([])))
    unknown = set(over) - set(a)
    assert not unknown, unknown
    a.update(over)
    return argparse.Namespace(**a)


def build(cls, data_dir, **over):
    solver = cls(make_args(data_dir, **over))
    solver.prepare()
    return solver


def rlsa_flip_prob(problem, graph, x, tau, d, return_logits=False):
    """The per-step lines of solvers/rlsa.py (RLSA.evaluate_single), verbatim, for state x."""
    _, grad = problem.energy_func(graph, x, True)
    delta = grad*(2*x-1)/2
    term2 = -torch.kthvalue(-delta, d, dim=0, keepdim=True).values
    flip_prob = torch.sigmoid((delta-term2)/tau)
    return ((delta-term2)/tau, flip_prob) if return_logits else flip_prob


class RecordRandLike:
    """Records the state passed to every torch.rand_like call (= the state before each flip)."""
    def __enter__(self):
        self.orig, self.states = torch.rand_like, []
        torch.rand_like = self._rand_like
        return self

    def _rand_like(self, x, *args, **kwargs):
        self.states.append(x.detach().clone())
        return self.orig(x, *args, **kwargs)

    def __exit__(self, *exc):
        torch.rand_like = self.orig


class TmpDir:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='camprior_')
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


# ---------------------------------------------------------------- (1) residual 0 => RLSA
def test_1a_flip_probs_equal_rlsa_per_step():
    """Zero residual: CAMPrior's per-step flip probs == RLSA's for identical states and temps."""
    T, K, d = 10, 8, 5
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [40, 31, 4], 0.15, seed=1)
        solver = build(CAMPrior, data, num_tp=T, num_kp=K, prior_d=d, prior_tau0=0.01)
        rlsa = build(RLSA, data, do_train=False, method='RLSA', num_t=T, num_k=K, num_d=d,
                     tau0=0.01, tau1=0.0, temp_schedule='linear')
        assert solver.prior_taus == [rlsa.temp_scheduler(k) for k in range(T)]
        gen = torch.Generator().manual_seed(0)
        n_checked = 0
        # single graphs: exactly RLSA (same kthvalue), bitwise
        for gi in (0, 1):
            graph = load_batch(data, [gi])
            for _ in range(3):
                x = torch.randint(0, 2, (graph.num_nodes, K), generator=gen).float()
                time_in = torch.zeros_like(x)
                for k in range(T):
                    for ratio in (1.0, 0.37):
                        logits, prior = solver.flip_logits(graph, x, time_in, k, ratio)
                        assert torch.equal(logits, prior), 'residual must be exactly 0 at init'
                        ref = rlsa_flip_prob(rlsa.problem, graph, x, rlsa.temp_scheduler(k)*ratio, d)
                        assert torch.equal(torch.sigmoid(logits), ref), (gi, k, ratio)
                        n_checked += 1
        # a batch of 3 graphs (one smaller than d): per-graph threshold == RLSA on each graph
        # with d clamped to the graph size. The logits are compared bitwise; the probabilities
        # only up to float32 rounding, because CPU sigmoid on a longer (concatenated) tensor can
        # take a different vectorised path and differ in the last bit.
        graph = load_batch(data)
        sizes = torch.bincount(graph.batch).tolist()
        x = torch.randint(0, 2, (graph.num_nodes, K), generator=gen).float()
        max_dp = 0.0
        for k in range(T):
            logits, _ = solver.flip_logits(graph, x, torch.zeros_like(x), k, 1.0)
            ref_logits, ref_probs = [], []
            for gi, xi in enumerate(torch.split(x, sizes)):
                lg, pr = rlsa_flip_prob(rlsa.problem, load_batch(data, [gi]), xi,
                                        rlsa.temp_scheduler(k), min(d, sizes[gi]), return_logits=True)
                ref_logits.append(lg)
                ref_probs.append(pr)
            assert torch.equal(logits, torch.cat(ref_logits)), k
            dp = (torch.sigmoid(logits) - torch.cat(ref_probs)).abs().max().item()
            assert dp <= 2.4e-7, (k, dp)
            max_dp = max(max_dp, dp)
        # --prior_beta: the prior uses its own energy, the training target keeps --beta
        solver_b = build(CAMPrior, data, num_tp=T, num_kp=K, prior_d=d, prior_beta=1.001)
        assert solver_b.prior_problem is not solver_b.problem and solver_b.problem.beta == 1.02
        rlsa_b = build(RLSA, data, do_train=False, method='RLSA', num_t=T, num_k=K, num_d=d,
                       tau0=0.01, beta=1.001)
        graph = load_batch(data, [0])
        x = torch.randint(0, 2, (graph.num_nodes, K), generator=gen).float()
        logits, _ = solver_b.flip_logits(graph, x, torch.zeros_like(x), 3, 1.0)
        assert torch.equal(torch.sigmoid(logits), rlsa_flip_prob(rlsa_b.problem, graph, x, rlsa_b.temp_scheduler(3), d))
    return (f'{n_checked} single-graph (state, step, ratio) cases bitwise equal; batch of 3 graphs '
            f'(one with n < d): logits bitwise equal, max|dp| {max_dp:.1e}; prior_beta ok')


def test_1b_rollout_equals_rlsa_chain():
    """Same seed, zero residual, epoch-0 temperature: CAMPrior's chains are RLSA's chains."""
    T, K, d = 12, 6, 5
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [45], 0.15, seed=7)
        solver = build(CAMPrior, data, num_tp=T, num_kp=K, num_t=T, num_k=K, prior_d=d)
        rlsa = build(RLSA, data, do_train=False, method='RLSA', num_t=T, num_k=K, num_d=d, tau0=0.01)
        solver.tau = solver.temp_scheduler(0)                    # epoch 0
        assert solver.prior_tau_ratio() == 1.0
        for seed in (0, 1, 2):
            graph = load_batch(data)
            torch.manual_seed(seed)
            with RecordRandLike() as rec_rlsa:
                rlsa.evaluate_single(graph)
            graph = load_batch(data)
            torch.manual_seed(seed)
            with RecordRandLike() as rec_train:
                solver.generate_data(graph)                      # training rollout
            graph = load_batch(data)
            torch.manual_seed(seed)
            with RecordRandLike() as rec_inf:
                solver.evaluate_single(graph)                    # inference rollout (last step soft)
            assert len(rec_rlsa.states) == T and len(rec_train.states) == T and len(rec_inf.states) == T - 1
            for t in range(T):
                assert torch.equal(rec_rlsa.states[t], rec_train.states[t]), (seed, t)
            for t in range(T - 1):
                assert torch.equal(rec_rlsa.states[t], rec_inf.states[t]), (seed, t)
            n_changed = sum(int((rec_rlsa.states[t] != rec_rlsa.states[t+1]).sum()) for t in range(T-1))
            assert n_changed > 0
    return f'3 seeds x {T} steps x {K} chains: identical states (training and inference rollouts)'


# ---------------------------------------------------------------- (2) p_ref = 1/2 => CAM loss + const
def _random_batch(graph, mb, gen, dtype=torch.float32):
    n = graph.num_nodes
    solution = torch.randint(0, 2, (mb, n), generator=gen).to(dtype)
    time_in = torch.rand(mb, n, generator=gen).to(dtype)
    weight = (0.2 + 0.8*torch.rand(mb, 1, generator=gen)).expand(mb, n).to(dtype).contiguous()
    target = (2*torch.randn(mb, n, generator=gen)).to(dtype)
    return solution, time_in, weight, target


def _grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


def test_2_uniform_prior_equals_cam_loss_plus_constant():
    """prior logit == 0: new loss == CAM's loss (solvers/cam.py) + tau*log2 per coordinate."""
    tau, lambd, num_d, mb = 0.07, 0.3, 4.0, 16
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [30, 22], 0.2, seed=3)
        solver = build(CAMPrior, data, residual_init='default')   # non-zero residual
        solver.tau, solver.lambd, solver.num_d = tau, lambd, num_d
        model = solver.model
        cam_self = types.SimpleNamespace(model=model, solution_type='node', tau=tau, lambd=lambd,
                                         num_d=num_d, loss_func=nn.BCEWithLogitsLoss())
        gen = torch.Generator().manual_seed(0)
        out = []
        for idxs in ([0], [0, 1]):                                 # one graph, and a batch of two
            graph = load_batch(data, idxs)
            sol, tim, w, tgt = _random_batch(graph, mb, gen)
            model.zero_grad()
            loss_cam = CAM.train_batch(cam_self, graph, [sol, tim, w, tgt])
            loss_cam.backward()
            g_cam = _grads(model)
            model.zero_grad()
            loss_new = CAMPrior.train_batch(solver, graph, [sol, tim, w, tgt, torch.zeros_like(tgt)])
            loss_new.backward()
            g_new = _grads(model)
            sizes = torch.bincount(graph.batch).double()
            const = tau*math.log(2)*sizes.mean().item()           # g_sum per graph, then mean
            # exact identity against CAM's loss with the exact entropy (no +1e-6 inside the logs)
            with torch.no_grad():
                logits = model(graph, sol.T.unsqueeze(-1), tim.T).squeeze(-1)
                from utils.graph_ops import g_sum
                p = torch.sigmoid(logits)
                neg_ent = p*torch.nn.functional.logsigmoid(logits) + (1-p)*torch.nn.functional.logsigmoid(-logits)
                loss_cam_exact = (-g_sum(p*tgt.T*torch.exp(-(1-w.T)), graph.batch).mean()
                                  + tau*g_sum(neg_ent, graph.batch).mean()
                                  + lambd*((g_sum(p, graph.batch)-num_d)**2).mean())
            err_exact = abs(loss_new.item() - loss_cam_exact.item() - const)
            err_cam = abs(loss_new.item() - loss_cam.item() - const)
            # CAM's +1e-6 inside the logs changes each coordinate's entropy by <= ~2e-6
            tol_cam = tau*sizes.max().item()*2e-6 + 1e-5
            assert err_exact < 1e-5, err_exact
            assert err_cam < tol_cam, (err_cam, tol_cam)
            gerr = max((a - b).abs().max().item() for a, b in zip(g_cam, g_new))
            gscale = max(a.abs().max().item() for a in g_cam)
            assert gerr < 1e-4*max(1.0, gscale), (gerr, gscale)
            out.append(f'{len(idxs)} graph(s): L_new-L_cam={loss_new.item()-loss_cam.item():.6f} '
                       f'vs tau*log2*n={const:.6f} (|diff| {err_cam:.1e}; exact-entropy |diff| '
                       f'{err_exact:.1e}); max|grad diff| {gerr:.1e} (grad scale {gscale:.2f})')
    return '; '.join(out)


# ---------------------------------------------------------------- (3) per-coordinate minimiser
class FreeResidual(nn.Module):
    """Stand-in for the network: one free residual logit per (node, sample)."""
    def __init__(self, n, mb, dtype):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(n, mb, dtype=dtype))

    def forward(self, graph, solution, time):
        return self.r.unsqueeze(-1)


def test_3_minimiser_is_prior_plus_target_over_tau():
    """argmin_p of the loss (lambd = 0) is sigmoid(prior + target*exp(-(1-w))/tau)."""
    mb = 5
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [12, 9], 0.3, seed=5)
        graph = load_batch(data)
        gen = torch.Generator().manual_seed(1)
        out = []
        for tau in (0.5, 0.1):
            sol, tim, w, tgt = _random_batch(graph, mb, gen, torch.float64)
            prior = (3*torch.randn(mb, graph.num_nodes, generator=gen)).double()
            stub = types.SimpleNamespace(model=FreeResidual(graph.num_nodes, mb, torch.float64),
                                         tau=tau, lambd=0.0, num_d=0.0, prior_policy=True,
                                         prior_kl_rlsa=True)
            batch = [sol, tim, w, tgt, prior]
            r_star = (tgt*torch.exp(-(1-w))/tau).T                 # analytic residual
            # (a) gradient of the real train_batch vanishes at the analytic minimiser
            with torch.no_grad():
                stub.model.r.copy_(r_star)
            stub.model.zero_grad()
            CAMPrior.train_batch(stub, graph, batch).backward()
            g_at_star = stub.model.r.grad.abs().max().item()
            assert g_at_star < 1e-12, g_at_star
            # (b) numerical minimisation from r = 0 lands there. Moderate magnitudes here: for
            # saturated coordinates (|prior + r*| >> 1) the loss is flat in r and LBFGS stops early.
            tgt_b, prior_b = 0.3*tgt, prior/3
            batch_b = [sol, tim, w, tgt_b, prior_b]
            r_star_b = (tgt_b*torch.exp(-(1-w))/tau).T
            with torch.no_grad():
                stub.model.r.zero_()
            opt = torch.optim.LBFGS(stub.model.parameters(), lr=1, max_iter=2000, tolerance_grad=1e-14,
                                    tolerance_change=1e-16, history_size=50, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = CAMPrior.train_batch(stub, graph, batch_b)
                loss.backward()
                return loss
            opt.step(closure)
            p_num = torch.sigmoid(prior_b.T + stub.model.r.detach())
            p_ana = torch.sigmoid(prior_b.T + r_star_b)
            err_p = (p_num - p_ana).abs().max().item()
            err_r = (stub.model.r.detach() - r_star_b).abs().max().item()
            assert err_p < 1e-6, err_p
            # (c) 1-D grid check on a few coordinates: loss along p is minimised at p*
            idx = [(0, 0), (3, 2), (graph.num_nodes - 1, mb - 1)]
            for (i, j) in idx:
                a, c = prior[j, i].item(), (tgt[j, i]*torch.exp(-(1-w[j, i]))).item()
                q = torch.sigmoid(torch.tensor(a, dtype=torch.float64))
                pg = torch.linspace(1e-6, 1-1e-6, 200001, dtype=torch.float64)
                f = -pg*c + tau*(pg*torch.log(pg/q) + (1-pg)*torch.log((1-pg)/(1-q)))
                p_grid = pg[f.argmin()].item()
                assert abs(p_grid - torch.sigmoid(torch.tensor(a + c/tau)).item()) < 2e-5
            out.append(f'tau={tau}: |grad| at analytic {g_at_star:.1e}, LBFGS max|p-p*| {err_p:.1e} '
                       f'(max|r-r*| {err_r:.1e}), grid ok')
    return '; '.join(out)


# ---------------------------------------------------------------- (4) tiny CPU training run
def test_4_tiny_rollout_backward_and_train_loop():
    """2 small graphs: rollout + backward give finite loss/grads; train/save/test round trip."""
    T, K = 6, 4
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [14, 18], 0.3, seed=11)
        save_dir = os.path.join(tmp, 'ckpt')
        solver = build(CAMPrior, data, num_tp=T, num_kp=K, num_t=T, num_k=K, prior_d=3, batch_size=2,
                       mini_batch_size=8, epochs=3, N_anneal=3, samples_per_epoch=2, save_dir=save_dir)
        head = solver.model.head
        assert head.weight.abs().sum().item() == 0 and head.bias.abs().sum().item() == 0
        # (a) one rollout on a batch of the two graphs, then a pass of mini-batch updates
        graph = next(iter(solver.train_loader))
        assert graph.num_graphs == 2
        graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight,
                                          torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
        solver.tau = solver.temp_scheduler(0)
        dataset = solver.generate_data(graph)
        assert len(dataset) == T*K and len(dataset[0]) == 5
        prior_all = torch.stack([item[4] for item in dataset])
        assert torch.isfinite(prior_all).all()
        losses = []
        solver.model.train()
        for b in DataLoader(dataset, batch_size=solver.mini_batch_size, shuffle=True):
            loss = solver.train_batch(graph, b)
            solver.optimizer.zero_grad()
            loss.backward()
            assert torch.isfinite(loss), loss
            for name, p in solver.model.named_parameters():
                assert p.grad is None or torch.isfinite(p.grad).all(), name
            losses.append(loss.item())
            if len(losses) == 1:
                assert head.weight.grad.abs().sum().item() > 0, 'zero head must still receive gradient'
            solver.optimizer.step()
        assert head.weight.abs().sum().item() > 0
        # (b) outer coupling: tau_prior / tau_rlsa = max(tau_train, tau_min) / tau_train(0)
        ratios = {}
        for tau in (0.1, 0.05, 1e-4, 0.0):
            solver.tau = tau
            ratios[tau] = solver.prior_tau_ratio()
        assert ratios[0.1] == 1.0 and abs(ratios[0.05] - 0.5) < 1e-12
        assert abs(ratios[1e-4] - 0.01) < 1e-12 and abs(ratios[0.0] - 0.01) < 1e-12
        solver.prior_fixed = True
        assert solver.prior_tau_ratio() == 1.0
        solver.prior_fixed = False
        # (c) the real training loop (3 epochs, validation, best.pt + sidecar); main.py makes save_dir
        os.makedirs(save_dir)
        solver.train(save_dir)
        ckpt = os.path.join(save_dir, 'best.pt')
        assert os.path.exists(ckpt) and os.path.exists(ckpt + PRIOR_META_SUFFIX)
        import json
        meta = json.load(open(ckpt + PRIOR_META_SUFFIX))
        sched = TempScheduler('linear', 0.1, 0.0, 0, 3, 0)
        assert any(abs(meta['tau_ratio'] - sched(e)/0.1) < 1e-12 for e in range(3)), meta
        # (d) test mode reads the outer factor from the sidecar and evaluates
        log = os.path.join(tmp, 'test', 'log.txt')
        os.makedirs(os.path.dirname(log))
        tester = build(CAMPrior, data, do_train=False, checkpoint=ckpt, num_t=T, num_k=K, num_tp=T,
                       prior_d=3)
        assert tester.prior_tau_ratio() == meta['tau_ratio']
        obj, _ = tester.evaluate('test', log_path=log)
        assert math.isfinite(obj) and obj < 0
        assert os.path.exists(log + '.per_graph.csv')
        # (d2) test-time prior settings that differ from the sidecar are an error unless allowed
        try:
            build(CAMPrior, data, do_train=False, checkpoint=ckpt, num_t=T, num_k=K, num_tp=T, prior_d=4)
            raise AssertionError('expected ValueError for a prior_d mismatch')
        except ValueError as e:
            assert 'prior_d' in str(e), e
        t_mm = build(CAMPrior, data, do_train=False, checkpoint=ckpt, num_t=T, num_k=K, num_tp=T, prior_d=4,
                     prior_allow_mismatch=True)
        assert t_mm.prior_tau_ratio() == meta['tau_ratio']
        # (e) without a sidecar, test mode refuses unless the scale is given
        bare = os.path.join(tmp, 'bare.pt')
        shutil.copy(ckpt, bare)
        try:
            build(CAMPrior, data, do_train=False, checkpoint=bare, num_tp=T, prior_d=3)
            raise AssertionError('expected FileNotFoundError')
        except FileNotFoundError:
            pass
        t2 = build(CAMPrior, data, do_train=False, checkpoint=bare, num_tp=T, prior_d=3, prior_tau_ratio=1.0)
        assert t2.prior_tau_ratio() == 1.0
    return (f'rollout {T}x{K} on 2 graphs, {len(losses)} updates, losses '
            f'{[round(v, 3) for v in losses]}; train loop 3 epochs ok, sidecar tau_ratio '
            f'{meta["tau_ratio"]:.3f}, test obj {obj:.2f}')


# ---------------------------------------------------------------- (5) ablation switches
def test_5_ablation_switches():
    """--prior_policy / --prior_kl_ref: both off == CAM (loss + const, grads, sampler); each one
    alone changes only its own place; the warm-start-only minimiser is CAM's, not prior + CAM's."""
    tau, lambd, num_d, mb = 0.07, 0.3, 4.0, 16
    from utils.graph_ops import g_sum
    F = torch.nn.functional
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [30, 22], 0.2, seed=7)
        graph = load_batch(data, [0, 1])
        gen = torch.Generator().manual_seed(2)
        sol, tim, w, tgt = _random_batch(graph, mb, gen)
        prior = 3*torch.randn(mb, graph.num_nodes, generator=gen)     # non-zero prior logits
        sizes = torch.bincount(graph.batch).double()
        out = []

        def exact_loss(logits, ref):
            p = torch.sigmoid(logits)
            kl = (p*(F.logsigmoid(logits) - F.logsigmoid(ref)) + (1-p)*(F.logsigmoid(-logits) - F.logsigmoid(-ref)))
            return (-g_sum(p*tgt.T*torch.exp(-(1-w.T)), graph.batch).mean()
                    + tau*g_sum(kl, graph.batch).mean() + lambd*((g_sum(p, graph.batch)-num_d)**2).mean())

        for policy, ref in (('off', 'uniform'), ('on', 'uniform'), ('off', 'rlsa'), ('on', 'rlsa')):
            solver = build(CAMPrior, data, residual_init='default', prior_policy=policy, prior_kl_ref=ref)
            solver.tau, solver.lambd, solver.num_d = tau, lambd, num_d
            model = solver.model
            model.zero_grad()
            loss = CAMPrior.train_batch(solver, graph, [sol, tim, w, tgt, prior])
            loss.backward()
            g_new = _grads(model)
            with torch.no_grad():
                res = model(graph, sol.T.unsqueeze(-1), tim.T).squeeze(-1)
                logits = res + prior.T if policy == 'on' else res
                refl = prior.T if ref == 'rlsa' else torch.zeros_like(res)
                err = abs(loss.item() - exact_loss(logits, refl).item())
            assert err < 1e-5, (policy, ref, err)
            if (policy, ref) == ('off', 'uniform'):
                # identical to CAM's own train_batch (up to tau*log2 per node and CAM's +1e-6)
                cam_self = types.SimpleNamespace(model=model, solution_type='node', tau=tau, lambd=lambd,
                                                 num_d=num_d, loss_func=nn.BCEWithLogitsLoss())
                model.zero_grad()
                loss_cam = CAM.train_batch(cam_self, graph, [sol, tim, w, tgt])
                loss_cam.backward()
                g_cam = _grads(model)
                const = tau*math.log(2)*sizes.mean().item()
                err_cam = abs(loss.item() - loss_cam.item() - const)
                assert err_cam < tau*sizes.max().item()*2e-6 + 1e-5, err_cam
                gerr = max((a - b).abs().max().item() for a, b in zip(g_cam, g_new))
                assert gerr < 1e-4*max(1.0, max(a.abs().max().item() for a in g_cam)), gerr
                # sampler: flip logits are the network output alone
                x = torch.randint(0, 2, (graph.num_nodes, 3), generator=gen).float()
                t_in = torch.zeros_like(x)
                with torch.no_grad():
                    fl, pr = solver.flip_logits(graph, x, t_in, 0, 1.0)
                    net = model(graph, x.unsqueeze(-1), t_in).squeeze(-1)
                assert torch.equal(fl, net) and pr.abs().max() > 0
                out.append(f'both off == CAM (|dL - const| {err_cam:.1e}, max|grad diff| {gerr:.1e})')
            else:
                out.append(f'policy {policy}/KL {ref}: loss matches formula ({err:.1e})')
            if ref == 'uniform':
                assert solver.prior_meta()['prior_kl_ref'] == 'uniform'

        # warm-start only (policy on, KL uniform): the gradient vanishes at r* = target/tau - prior,
        # i.e. p* = sigmoid(target/tau) does not depend on the prior
        stub = types.SimpleNamespace(model=FreeResidual(graph.num_nodes, mb, torch.float64), tau=0.3,
                                     lambd=0.0, num_d=0.0, prior_policy=True, prior_kl_rlsa=False)
        batch = [b.double() for b in (sol, tim, w, 0.3*tgt, prior/3)]
        with torch.no_grad():
            stub.model.r.copy_((batch[3]*torch.exp(-(1-batch[2]))/0.3 - batch[4]).T)
        CAMPrior.train_batch(stub, graph, batch).backward()
        g_star = stub.model.r.grad.abs().max().item()
        assert g_star < 1e-12, g_star
        out.append(f'warm-start-only minimiser is CAM\'s (|grad| {g_star:.1e})')
    return '; '.join(out)



def test_6_dense_labels():
    """--label_mode dense: lambda = 1 reproduces CAM's buffer exactly (same random numbers); other
    lambdas give sum_j w_j grad(X_j) * (2 x_t - 1) with the explicit weights of the docstring."""
    out = []
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [26, 19], 0.25, seed=11)
        graph = load_batch(data, [0, 1])
        T, K = 6, 4

        def gen(**over):
            torch.manual_seed(5)
            solver = build(CAMPrior, data, residual_init='default', **over)
            torch.manual_seed(9)
            return solver, solver.generate_data(graph)

        _, ref = gen()
        _, d1 = gen(label_mode='dense', label_lambda=1.0)
        assert len(ref) == len(d1) == T*K and len(d1[0]) == 6
        for a, b in zip(ref, d1):
            for c in range(5):
                assert torch.equal(a[c], b[c]), c
            assert torch.equal(b[5], torch.ones_like(b[5]))
        out.append('lambda 1 == CAM buffer')

        for lam in (0.0, 0.5, 0.8):
            solver, ds = gen(label_mode='dense', label_lambda=lam)
            X = torch.stack([ds[i][0] for i in range(T*K)]).reshape(K, T, -1)        # chain, step, node
            assert torch.equal(torch.stack([ds[i][0] for i in range(T*K)])[:T], X[0])
            G = torch.stack([solver.problem.energy_func(graph, X[:, t].T, True)[1].T for t in range(T)], 1)
            err = 0.0
            for t in range(T):
                if t == T - 1:
                    w = {T - 1: 1.0}
                else:
                    w = {j: (1 - lam)*lam**(j - t - 1) for j in range(t + 1, T - 1)}
                    w[T - 1] = lam**(T - 2 - t)
                assert abs(sum(w.values()) - 1) < 1e-12
                R = sum(wj*G[:, j] for j, wj in w.items())
                expect = R*(2*X[:, t] - 1)
                got = torch.stack([ds[k*T + t][3] for k in range(K)])
                err = max(err, (got - expect).abs().max().item())
            assert err < 1e-5, (lam, err)
            out.append(f'lambda {lam}: matches formula ({err:.1e})')
        try:
            build(CAMPrior, data, label_mode='dense', label_lambda=0.5, num_clips=2)
            raise AssertionError('num_clips 2 accepted')
        except NotImplementedError:
            pass
    return '; '.join(out)



def test_7_nosign_labels():
    """--label_mode nosign: CAM's target times (1 - 2 p), p = node flips an odd number of times after
    step t; same rollout as CAM (identical states, time, prior columns)."""
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [26, 19], 0.25, seed=13)
        graph = load_batch(data, [0, 1])
        T, K = 6, 4

        def gen(**over):
            torch.manual_seed(5)
            solver = build(CAMPrior, data, residual_init='default', **over)
            torch.manual_seed(9)
            return solver, solver.generate_data(graph)

        _, ref = gen()
        solver, ns = gen(label_mode='nosign')
        X = torch.stack([ns[i][0] for i in range(T*K)]).reshape(K, T, -1)            # chain, step, node
        for a, b in zip(ref, ns):
            for c in (0, 1, 2, 4):
                assert torch.equal(a[c], b[c]), c
        # the state after the last step is not stored; p only needs X_{t+1} for t < T - 1
        n_odd, err = 0, 0.0
        for k in range(K):
            for t in range(T):
                cam_t, ns_t = ref[k*T + t][3], ns[k*T + t][3]
                odd = (X[k, T - 1] != X[k, t + 1]).float() if t < T - 1 else torch.zeros_like(cam_t)
                n_odd += int(odd.sum())
                err = max(err, (ns_t - cam_t*(1 - 2*odd)).abs().max().item())
        assert err < 1e-6 and n_odd > 0, (err, n_odd)
    return f'nosign == CAM * (1 - 2p) ({n_odd} sign flips, max err {err:.1e})'


def test_8_simpol_labels():
    """--label_mode simpol: (a) with a zero residual and no KL it equals the sim label (RLSA
    rollouts, same uniforms); (b) with a random residual each label equals a direct rollout of the
    current sampler from the factual and the toggled next state, with the same uniforms, including
    the tau * KL cost along the rollout."""
    from solvers.sim_labels import item_uniforms
    from solvers.cam_prior import bernoulli_kl_logits
    out = []
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [23], 0.25, seed=13)
        graph = load_batch(data, [0])
        T, K = 6, 4

        def gen(**over):
            torch.manual_seed(5)
            solver = build(CAMPrior, data, label_frac=0.2, **over)
            solver.tau = solver.temp_scheduler(0)
            torch.manual_seed(9)
            return solver, solver.generate_data(graph)

        _, sim = gen(label_mode='sim')
        _, pol = gen(label_mode='simpol', simpol_kl='off')
        err = max((a[3] - b[3]).abs().max().item() for a, b in zip(sim, pol))
        nz = sum(int((b[3] != 0).sum()) for b in pol)
        assert err < 1e-5 and nz > 0, (err, nz)
        out.append(f'zero residual == sim ({nz} labels, max err {err:.1e})')

        # --label_states: the labelled states keep exactly the labels of the full run
        _, full = gen(label_mode='simpol')
        _, part = gen(label_mode='simpol', label_states=0.25)
        kept = [i for i, b in enumerate(part) if b[5].sum() > 0]
        assert len(kept) == round(0.25*T*K), len(kept)
        for i, (a, b) in enumerate(zip(full, part)):
            if i in kept:
                assert torch.equal(a[3], b[3]) and torch.equal(a[5], b[5]), i
            else:
                assert b[3].abs().sum() == 0 and b[5].sum() == 0, i
        out.append(f'label_states 0.25: {len(kept)} states keep the full-run labels')

        torch.manual_seed(3)
        solver = build(CAMPrior, data, residual_init='default', label_mode='simpol')
        solver.tau = 0.07
        n = graph.num_nodes
        gen_ = torch.Generator().manual_seed(1)
        C, m = 3, 2
        x_next = torch.randint(0, 2, (n, C), generator=gen_).float()
        sel = torch.stack([torch.randperm(n, generator=gen_)[:m] for _ in range(C)])
        sign = torch.tensor([[1., -1.], [-1., 1.], [1., 1.]])
        starts, seeds, ratio = [4, 2, T], [11, 22, 33], 0.8
        got = solver.policy_cf_labels(graph, x_next, sel, sign, starts, seeds, ratio)
        err = 0.0
        for c in range(C):
            U = item_uniforms(torch.tensor([seeds[c]]), torch.tensor([starts[c]]), T, n)[0]
            X = x_next[:, c:c + 1].repeat(1, m + 1)
            for i in range(m):
                X[sel[c, i], i + 1] = 1 - X[sel[c, i], i + 1]
            kl = torch.zeros(m + 1)
            for j in range(starts[c], T):
                logits, prior = solver.flip_logits(graph, X, torch.ones_like(X)*j/T, j, ratio)
                kl += 0.07*bernoulli_kl_logits(logits, prior).sum(0)
                X = torch.where(U[j].unsqueeze(1) < torch.sigmoid(logits), 1 - X, X)
            cost = solver.problem.energy_func(graph, X, False)[0].reshape(-1) + kl
            expect = sign[c]*(cost[0] - cost[1:])
            err = max(err, (got[c] - expect).abs().max().item())
        assert err < 1e-4, err
        out.append(f'random residual == direct rollout incl. KL (max err {err:.1e})')
    return '; '.join(out)


# ---------------------------------------------------------------- runner
if __name__ == '__main__':
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            info = fn()
            print(f'PASS {name} ({time.time()-t0:.1f}s): {info}', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name} ({time.time()-t0:.1f}s)', flush=True)
            traceback.print_exc()
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
