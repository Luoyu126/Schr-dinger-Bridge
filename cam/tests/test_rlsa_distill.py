"""Tests for the distillation warm start: solvers/rlsa_distill.py, distill_rlsa.py and the
launchers scripts/mis/mis_er700_distill.sh and scripts/mis/distill_warmstart_er700.sh. CPU only.

Run from the repository root with the CAM env (own runner; the test_* functions also run
under pytest):

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_rlsa_distill.py
"""
import os
import re
import subprocess
import sys
import time
import traceback
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU only, before torch is imported
os.environ.setdefault('WANDB_MODE', 'disabled')
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS_DIR)
import test_cam_prior as tcp  # noqa: E402  (helpers; sets sys.path and the temp dir)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from solvers.cam import CAM  # noqa: E402
from solvers.cam_prior import CAMPrior, PRIOR_META_SUFFIX  # noqa: E402
from solvers.rlsa import RLSA  # noqa: E402
from solvers.rlsa_distill import RLSADistill, distill_loss  # noqa: E402
from utils.graph_ops import g_sum  # noqa: E402
import distill_rlsa  # noqa: E402

CAM_ROOT, REPO_ROOT = tcp.CAM_ROOT, tcp.REPO_ROOT
TmpDir, build, load_batch, write_dataset = tcp.TmpDir, tcp.build, tcp.load_batch, tcp.write_dataset


# ---------------------------------------------------------------- (1) teacher = RLSA
def test_1_teacher_rollout_is_rlsa():
    """Same seed: the stored states are RLSA's chain (= CAMPrior's untrained chain) and the stored
    teacher logits are RLSA's flip logits at those states, with prior tau0 = 1 as in the p1 runs."""
    T, K, d, tau0 = 10, 6, 5, 1.0
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [45], 0.15, seed=7)
        dist = build(RLSADistill, data, num_tp=T, num_kp=K, prior_d=d, prior_tau0=tau0)
        cp = build(CAMPrior, data, num_tp=T, num_kp=K, prior_d=d, prior_tau0=tau0)   # zero head
        rlsa = build(RLSA, data, do_train=False, method='RLSA', num_t=T, num_k=K, num_d=d, tau0=tau0,
                     tau1=0.0, temp_schedule='linear')
        cp.tau = cp.temp_scheduler(0)
        assert dist.prior_tau_ratio() == 1.0 and cp.prior_tau_ratio() == 1.0
        n = 0
        for seed in (0, 1, 2):
            graph = load_batch(data)
            torch.manual_seed(seed)
            with tcp.RecordRandLike() as rec:
                ds = dist.generate_data(graph)
            graph = load_batch(data)
            torch.manual_seed(seed)
            ds_cp = cp.generate_data(graph)
            graph = load_batch(data)
            torch.manual_seed(seed)
            with tcp.RecordRandLike() as rec_rlsa:
                rlsa.evaluate_single(graph)
            assert len(ds) == T*K and len(ds[0]) == 3
            # identical to CAMPrior's untrained rollout: states, time inputs and prior logits
            for a, b in zip(ds, ds_cp):
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[4])
            # the visited states are RLSA's chain
            for t in range(T):
                assert torch.equal(rec.states[t], rec_rlsa.states[t]), (seed, t)
            # teacher logits = RLSA's (delta - delta_(d)) / tau_rlsa(k) at the stored state
            states = torch.stack([c[0] for c in ds], 1).view(-1, K, T)
            teach = torch.stack([c[2] for c in ds], 1).view(-1, K, T)
            times = torch.stack([c[1] for c in ds], 1).view(-1, K, T)
            for t in range(T):
                lg, _ = tcp.rlsa_flip_prob(rlsa.problem, graph, states[:, :, t], rlsa.temp_scheduler(t), d,
                                           return_logits=True)
                assert torch.equal(teach[:, :, t], lg), (seed, t)
                assert torch.all(times[:, :, t] == t/T)
                n += 1
    return f'3 seeds x {T} steps x {K} chains: states = RLSA chain = CAMPrior chain, teacher = RLSA logits ({n} checks)'


# ---------------------------------------------------------------- (2) loss
class FreeLogits(nn.Module):
    def __init__(self, n, mb):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(n, mb, dtype=torch.float64))

    def forward(self, graph, solution, time):
        return self.r.unsqueeze(-1)


def test_2_loss_is_forward_kl():
    """0 at the teacher, > 0 elsewhere; gradient sigmoid(s) - sigmoid(a) (over graphs x columns);
    a free student fitted by LBFGS reproduces the teacher's flip probabilities."""
    mb = 7
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [20, 13], 0.25, seed=3)
        graph = load_batch(data)
        gen = torch.Generator().manual_seed(0)
        n = graph.num_nodes
        a = (4*torch.randn(n, mb, generator=gen)).double()
        a[0, 0], a[1, 1] = 40.0, -40.0                            # saturated teacher coordinates
        assert distill_loss(a.clone(), a, graph.batch).abs().item() < 1e-12
        s = (a + torch.randn(n, mb, generator=gen).double()).requires_grad_(True)
        loss = distill_loss(s, a, graph.batch)
        assert loss.item() > 0
        loss.backward()
        expect = (torch.sigmoid(s) - torch.sigmoid(a)).detach() / (graph.num_graphs*mb)
        gerr = (s.grad - expect).abs().max().item()
        assert gerr < 1e-12, gerr
        # train_batch with a free student: minimiser = teacher (moderate logits, as LBFGS needs)
        a_b = a.clamp(-6, 6)
        stub = types.SimpleNamespace(model=FreeLogits(n, mb))
        batch = [torch.zeros(mb, n).double(), torch.zeros(mb, n).double(), a_b.T]
        opt = torch.optim.LBFGS(stub.model.parameters(), lr=1, max_iter=500, tolerance_grad=1e-14,
                                tolerance_change=1e-16, history_size=50, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            out = RLSADistill.train_batch(stub, graph, batch)
            out.backward()
            return out
        opt.step(closure)
        err = (torch.sigmoid(stub.model.r.detach()) - torch.sigmoid(a_b)).abs().max().item()
        assert err < 1e-6, err
    return f'KL(teacher || student): 0 at teacher, |grad - (p_s - p_t)/(G*B)| {gerr:.1e}, LBFGS max|dp| {err:.1e}'


# ---------------------------------------------------------------- (3) student = CAM sampler
def test_3_student_is_plain_cam():
    """RLSA enters neither the student's logits nor its initialisation (even if the CAMPrior flags
    ask for it), and validation is CAM's own inference with the same weights."""
    T, K = 8, 5
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [30, 26], 0.2, seed=5)
        dist = build(RLSADistill, data, num_tp=T, num_kp=K, num_t=T, num_k=K, prior_d=4, prior_tau0=1.0,
                     prior_policy='on', residual_init='zero', prior_fixed=False)
        assert not dist.prior_policy and dist.prior_fixed
        assert dist.args.residual_init == 'default' and dist.model.head.weight.abs().sum().item() > 0
        graph = load_batch(data)
        x = torch.randint(0, 2, (graph.num_nodes, K), generator=torch.Generator().manual_seed(1)).float()
        t_in = torch.full_like(x, 3/T)
        with torch.no_grad():
            fl, pr = dist.flip_logits(graph, x, t_in, 3, 1.0)
            net = dist.model(graph, x.unsqueeze(-1), t_in).squeeze(-1)
        assert torch.equal(fl, net) and pr.abs().max() > 0
        # CAM with the same weights gives the same inference result, step for step
        cam = build(CAM, data, method='CAM', num_tp=T, num_kp=K, num_t=T, num_k=K)
        cam.model.load_state_dict(dist.model.state_dict())
        res = []
        for solver in (dist, cam):
            solver.model.eval()
            graph = load_batch(data, [0])
            torch.manual_seed(4)
            with torch.no_grad(), tcp.RecordRandLike() as rec:
                obj = solver.evaluate_single(graph)
            res.append((obj, rec.states))
        assert torch.equal(res[0][0], res[1][0])
        assert all(torch.equal(u, v) for u, v in zip(res[0][1], res[1][1])) and len(res[0][1]) == T - 1
    return f'student logits = network only, CAM init; evaluate_single == CAM.evaluate_single (obj {res[0][0].item():.0f})'


# ---------------------------------------------------------------- (4) training loop -> CAM
def test_4_train_loop_and_cam_loads_checkpoint():
    """Real training loop: loss falls, plain CAM checkpoints (no sidecar), and CAM training
    started with --checkpoint holds exactly these weights with a fresh optimiser and schedule."""
    T, K = 6, 4
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [16, 20], 0.25, seed=11)
        save_dir = os.path.join(tmp, 'ckpt')
        dist = build(RLSADistill, data, num_tp=T, num_kp=K, num_t=T, num_k=K, prior_d=3, prior_tau0=1.0,
                     batch_size=1, mini_batch_size=8, epochs=6, N_anneal=6, samples_per_epoch=2, lr=3e-3,
                     save_dir=save_dir)
        dist.args.save_every_epoch = True
        logged = []
        dist.log = lambda d: logged.append(d)
        os.makedirs(save_dir)
        torch.manual_seed(0)
        dist.train(save_dir)
        losses = [d['train/loss'] for d in logged if 'train/loss' in d]
        assert len(losses) == 6 and losses[-1] < losses[0], losses
        files = sorted(os.listdir(save_dir))
        assert 'best.pt' in files and 'epoch_init.pt' in files and 'epoch_0005.pt' in files, files
        assert not any(f.endswith(PRIOR_META_SUFFIX) for f in files), files
        ckpt = os.path.join(save_dir, 'best.pt')
        cam = build(CAM, data, method='CAM', num_tp=T, num_kp=K, checkpoint=ckpt, epochs=3, N_anneal=3)
        saved = torch.load(ckpt, weights_only=True)
        assert set(saved) == set(cam.model.state_dict())
        assert all(torch.equal(saved[k], v) for k, v in cam.model.state_dict().items())
        assert cam.global_step == 0 and cam.tau == 0.1
        assert all(len(st) == 0 for st in cam.optimizer.state.values())
    return f'6 epochs, train loss {losses[0]:.3f} -> {losses[-1]:.3f}; best.pt loads into CAM (fresh optimiser, tau 0.1)'


# ---------------------------------------------------------------- (5) entry point and launchers
def test_5_entry_point_and_launchers():
    """distill_rlsa.py runs main.py's code with RLSADistill; the stage-1 launcher is CAM's with
    main.py replaced; stage 2 is exactly the CAM tau0=0.1 command of seeds_er700.sh + --checkpoint."""
    # (a) the entry point insists on --method CAMPrior
    for argv in ([], ['--method', 'CAM']):
        try:
            distill_rlsa.run(argv)
            raise AssertionError('expected SystemExit')
        except SystemExit as e:
            assert 'CAMPrior' in str(e)
    # (b) end to end in-process on two tiny graphs: an RLSADistill run (no CAMPrior sidecar)
    with TmpDir() as tmp:
        data = write_dataset(os.path.join(tmp, 'g'), [12, 15], 0.3, seed=2)
        save_dir = os.path.join(tmp, 'run')
        argv = ['--train_path', data, '--valid_path', data, '--test_path', data, '--problem', 'mis',
                '--samples_per_epoch', '2', '--valid_sample', '2', '--batch_size', '1', '--mini_batch_size', '8',
                '--num_t', '4', '--num_k', '2', '--num_tp', '4', '--num_kp', '2', '--num_h', '8', '--num_l', '2',
                '--method', 'CAM', '--model', 'EncodeProcessDecode', '--epochs', '1000', '--N_anneal', '1000',
                '--lr', '0.0005', '--save_dir', save_dir, '--wandb_name', 't', '--wandb_run_name', 't',
                '--do_train', '--num_workers', '0',
                '--method', 'CAMPrior', '--prior_tau0', '1', '--prior_d', '3', '--epochs', '1', '--N_anneal', '1',
                '--save_every_epoch', '--seed', '0']
        argv0, cls0 = sys.argv, distill_rlsa.main.CAMPrior
        try:
            distill_rlsa.run(argv)
            assert distill_rlsa.main.CAMPrior is RLSADistill
        finally:
            sys.argv, distill_rlsa.main.CAMPrior = argv0, cls0
        files = sorted(os.listdir(save_dir))
        assert {'best.pt', 'epoch_init.pt', 'epoch_0000.pt'} <= set(files), files
        assert not any(f.endswith(PRIOR_META_SUFFIX) for f in files), files
        init = torch.load(os.path.join(save_dir, 'epoch_init.pt'), weights_only=True)
        assert init['head.weight'].abs().sum().item() > 0          # CAM init, not the zero head
    # (c) stage-1 launcher: CAM's launch block with main.py -> distill_rlsa.py, nothing else
    mis = os.path.join(CAM_ROOT, 'scripts', 'mis')
    block = lambda p: open(p).read().partition('cd "$CAM_ROOT"\n')[2]
    cam_block, dist_block = block(os.path.join(mis, 'mis_er700_cam.sh')), block(os.path.join(mis, 'mis_er700_distill.sh'))
    assert cam_block and dist_block == cam_block.replace(' main.py \\\n', ' distill_rlsa.py \\\n') != cam_block
    # (d) stage 2 == the CAM tau0=0.1 command of seeds_er700.sh plus --checkpoint (DRY runs)
    seed = '97'
    env = dict(os.environ, DRY='1', SEEDS=seed)
    out_cam = subprocess.run(['bash', os.path.join(mis, 'seeds_er700.sh')], env=dict(env, GPUS='8 9', SET='seeds'),
                             capture_output=True, text=True, check=True).stdout
    out_dis = subprocess.run(['bash', os.path.join(mis, 'distill_warmstart_er700.sh')], env=dict(env, GPUS='8'),
                             capture_output=True, text=True, check=True).stdout
    cam_cmd = re.search(r'^\s+(CUDA_VISIBLE_DEVICES=8 .*EXP=CAM_tau0\.1_e100_s97 .*)$', out_cam, re.M).group(1)
    st1 = re.search(r'stage 1: (.*)$', out_dis, re.M).group(1)
    st2 = re.search(r'stage 2: (.*)$', out_dis, re.M).group(1)
    ckpt = f"'{REPO_ROOT}/runs/cam/mis_er700/RLSA_distill_p1_s{seed}/best.pt'"
    assert st2 == cam_cmd.replace('EXP=CAM_tau0.1_e100_s97', 'EXP=CAM_tau0.1_distill_e100_s97') + ' --checkpoint ' + ckpt, (st2, cam_cmd)
    assert 'mis_er700_distill.sh --method CAMPrior --prior_tau0 1 --epochs 20 --N_anneal 20' in st1
    assert f'--save_every_epoch --seed {seed}' in st1 and 'EXP=RLSA_distill_p1_s97' in st1
    return 'entry point ok (run end to end); stage-1 launcher = CAM launcher with main.py swapped; stage 2 = CAM tau0=0.1 seeds command + --checkpoint'


# ---------------------------------------------------------------- runner
if __name__ == '__main__':
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)
             and getattr(v, '__module__', None) == '__main__']
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
