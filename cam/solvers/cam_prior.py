"""CAM with an RLSA energy prior as the base process (local addition, see cam/UPSTREAM.md).

CAMPrior is CAM (solvers/cam.py) with CAM's uniform base process replaced by an analytic,
energy-based one: the flip kernel of RLSA (solvers/rlsa.py; RLD paper, Algorithm 1 and Eq. 14).
The network only learns a residual on top of it.

Sampler (training rollout: num_tp steps, num_kp chains; inference: num_t steps, num_k chains)
    X_0 ~ uniform{0,1}^N
    p_i(x, k)  = sigmoid( a_i(x, k) + r_theta(x)_i )                 per-coordinate flip prob
    a_i(x, k)  = (delta_i - delta_(d) + eps) / tau_prior(k, epoch)   prior (RLSA) flip logit
    delta      = g * (2x - 1) / 2,  g = grad of the penalty energy (envs/<problem>.py)
    delta_(d)  = d-th largest entry of delta in the graph, per chain (RLSA's kthvalue)
r_theta is CAM's network (EncodeProcessDecode on MIS) with its output head zero-initialised
(--residual_init zero, the default), so r_theta == 0 at initialisation. g, delta, the top-d
threshold and the inner schedule are computed exactly as in solvers/rlsa.py, so with
r_theta == 0 and tau_prior = tau_rlsa the per-step flip probabilities are RLSA's
(tests/test_cam_prior.py checks this for identical states, and that the chains coincide).
RLSA's code uses no eps (the paper's optional eps >= 0 is in Delta = 2*delta units, i.e. it
equals 2 * --prior_eps here); --prior_eps defaults to 0 like the code.

Temperature of the prior: two time axes (docs/plan.md, section 0.8)
    tau_prior(k, epoch) = tau_rlsa(k) * max(tau_train(epoch), tau_min) / tau_train(0)
  - INNER, along the sampling steps k = 0..num_tp-1 of one chain: RLSA's schedule
    tau_rlsa(k) = TempScheduler(--prior_schedule, --prior_tau0, --prior_tau1, 0, num_tp, 0)(k),
    i.e. prior_tau0 * (1 - k/num_tp) for the default linear schedule with prior_tau1 = 0.
    At inference with num_t != num_tp the step index is mapped like CAM maps its time input:
    k = int(t * num_tp / num_t).
  - OUTER, along training: the prior logit scales like 1/tau_train, CAM's training temperature
    (--tau0/--tau1/--N_anneal ...). Written as "prior logit = kappa * (...) / tau_train":
        a_i = kappa * (delta_i - delta_(d) + eps) / ((tau_rlsa(k) / prior_tau0) * tau_train(epoch)),
        kappa = tau_train(0) / prior_tau0,
    kappa is chosen so that at epoch 0 the prior is exactly RLSA with its tuned tau0
    (e.g. tau_train(0) = 0.1 and prior_tau0 = 0.01 give kappa = 10). This mirrors CAM's own base
    rate sigma(-lambda/tau) (CAM paper, Eq. 10), whose logit also scales like 1/tau.
    tau_train -> 0 is made safe by clamping it at --prior_tau_min (default 1e-3, i.e. at most a
    100x amplification for tau_train(0) = 0.1; with tau1 = 0 the clamp only matters in the last
    epochs of the anneal and in an equilibrium phase).
    --prior_fixed is the ablation without the outer coupling (tau_prior = tau_rlsa(k) always).

Loss: CAM's loss (solvers/cam.py, node case) with the entropy term -tau*H(p) replaced by the
KL to the prior's Bernoulli, tau * KL(Ber(p) || Ber(p_ref)), p_ref = sigmoid(a) (detached);
the adjoint term and the optional lambd term are unchanged:
    L = -sum_i p_i * target_i * exp(-(1 - w)) + tau_train * sum_i KL_i + lambd * (sum_i p_i - num_d)^2
This is the per-step discretisation of the KL to the base process in CAM's control objective
(CAM paper, Eqs. 4-5 and 29-30). With a = 0 (p_ref = 1/2) it equals CAM's loss plus the
constant tau * N * log 2. For lambd = 0 the per-coordinate minimiser is
    p* = sigmoid(a + target * exp(-(1 - w)) / tau_train),
so the residual learns target * exp(-(1 - w)) / tau_train (CAM's own optimum is a = 0).

Inference: the same rollout with the same prior (inner schedule mapped to num_t steps; outer
scale = the one the checkpoint was trained with), last step soft, then CAM's greedy decoding.
The outer scale of a checkpoint is saved next to it as <checkpoint>.prior.json and read at
test time; --prior_tau_ratio overrides it (1.0 = the epoch-0 / RLSA scale).
Only node-based problems are supported (like RLSA).

Ablations (docs/status.md): the prior enters CAMPrior in two places, and each can be switched off.
    --prior_policy off     flip logits = r_theta alone (the prior only enters the KL); use with
                           --residual_init default, i.e. CAM's own initialisation
    --prior_kl_ref uniform KL to Ber(1/2) instead of Ber(sigmoid(a)), i.e. CAM's entropy term
                           (the prior only enters the policy: a warm start at RLSA whose optimum
                           p* = sigmoid(target * exp(-(1 - w)) / tau_train) is CAM's, not RLSA's)
With both switched off the loss is CAM's loss plus a constant and the sampler is CAM's
(tests/test_cam_prior.py checks both).

Training labels (docs/status.md; 11-10 decision): only the label changes, the rest of training not.
    --label_mode cam       CAM's lean target (default)
    --label_mode sim       simulated label: RLSA rollouts from the two candidate next states
                           (solvers/sim_labels.py), on the GPUs of --label_gpus
    --label_mode lin       first-order label along the trajectory (solvers/lin_labels.py,
                           derivation in docs/label_correction.md); own perturbation only, unless
                           --lin_neighbors (adds the neighbour-reaction term, which diverges)
    --label_frac f         per stored state, round(f * N) nodes sampled uniformly enter the loss
                           (adjoint and KL terms, rescaled by N/m); f = 1 and cam = the original path
"""
import atexit
import copy
import importlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

from .cam import CAM
from .sim_labels import SimLabelPool, simulated_labels
from .lin_labels import linear_labels
from utils.data_utils import BatchBuffer
from utils.graph_ops import g_sum
from utils.temp_schedulers import TempScheduler

PRIOR_META_SUFFIX = '.prior.json'


def add_cam_prior_args(parser):
    """Command-line flags of CAMPrior (all new names, so existing flags parse as before)."""
    g = parser.add_argument_group('CAMPrior: CAM + RLSA energy prior (local addition)')
    g.add_argument('--prior_tau0', type=float, default=0.01,
                   help='RLSA initial temperature of the prior (RLD Table 4: 0.01)')
    g.add_argument('--prior_tau1', type=float, default=0.0, help='RLSA final temperature of the prior')
    g.add_argument('--prior_schedule', default='linear', choices=['linear', 'exp'],
                   help='inner (per-chain) schedule of the prior temperature over the num_tp steps')
    g.add_argument('--prior_d', type=int, default=20,
                   help="RLSA's regularised step size d of the prior (RLD Table 4, ER: 20)")
    g.add_argument('--prior_eps', type=float, default=0.0,
                   help='eps added to delta_i - delta_(d) (RLSA code: none)')
    g.add_argument('--prior_beta', type=float, default=None,
                   help='penalty coefficient of the energy used by the prior (default: --beta, '
                        'the training energy; RLD used 1.001 on ER)')
    g.add_argument('--prior_fixed', action='store_true',
                   help='ablation: prior temperature not coupled to the training temperature')
    g.add_argument('--prior_tau_min', type=float, default=1e-3,
                   help='lower clamp on tau_train in the outer coupling')
    g.add_argument('--prior_tau_ratio', type=float, default=None,
                   help='test time: outer factor tau_train/tau_train(0) to use; default: read from '
                        '<checkpoint>' + PRIOR_META_SUFFIX)
    g.add_argument('--prior_policy', default='on', choices=['on', 'off'],
                   help='ablation: off = flip logits are the network output alone (prior only in the KL)')
    g.add_argument('--prior_kl_ref', default='rlsa', choices=['rlsa', 'uniform'],
                   help="ablation: uniform = KL to Ber(1/2), i.e. CAM's entropy term (prior only in the policy)")
    g.add_argument('--prior_allow_mismatch', action='store_true',
                   help='test time: allow prior settings that differ from the checkpoint sidecar '
                        '(otherwise an error)')
    g.add_argument('--residual_init', default='zero', choices=['zero', 'default'],
                   help='zero-initialise the output head, so that the untrained sampler is RLSA')
    g.add_argument('--label_mode', default='cam', choices=['cam', 'sim', 'lin'],
                   help="training label: cam = CAM's lean adjoint target; sim = simulated label, "
                        'RLSA rollouts from the two candidate next states (solvers/sim_labels.py); '
                        'lin = first-order label along the trajectory (solvers/lin_labels.py)')
    g.add_argument('--lin_neighbors', action='store_true',
                   help='lin labels: add the neighbour-reaction term (NAAS Hessian analogue; the '
                        'recursion diverges, docs/label_correction.md 3.8)')
    g.add_argument('--label_frac', type=float, default=1.0,
                   help='fraction of nodes per stored state that enter the loss (sampled uniformly, '
                        'fresh per graph; the loss is rescaled by num_nodes/m)')
    g.add_argument('--label_gpus', default='',
                   help='physical GPU ids for the sim-label workers, e.g. 4,5,6,7 (empty: compute '
                        "in-process on the trainer's device)")
    g.add_argument('--label_timeout', type=float, default=900.0,
                   help='seconds after which a sim-label task counts as hung (worker restarted)')
    return parser


def zero_output_head(model):
    """Zero the final Linear layer (EncodeProcessDecode*.head or AnGNN.out); returns its name."""
    for name in ('head', 'out'):
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return name
    raise ValueError(f'{type(model).__name__} has no Linear output head (head/out) to zero-initialise')


def topd_threshold(delta, batch, num_graphs, d):
    """d-th largest entry of delta (num_nodes x K) within each graph, separately per column.

    For a single graph this is literally RLSA's -torch.kthvalue(-delta, d, dim=0, keepdim=True)
    (shape 1 x K, broadcasts); for a batch of graphs it is the same per graph (num_nodes x K).
    d is clamped to the number of nodes of each graph.
    """
    if num_graphs == 1:
        return -torch.kthvalue(-delta, min(d, delta.shape[0]), dim=0, keepdim=True).values
    dense, mask = to_dense_batch(delta, batch, fill_value=float('-inf'))  # B x Nmax x K
    top = torch.topk(dense, min(d, dense.shape[1]), dim=1).values           # B x d x K, descending
    kth = mask.sum(1).clamp(max=d) - 1                                       # B
    thr = top[torch.arange(dense.shape[0], device=delta.device), kth]       # B x K
    return thr[batch]


def bernoulli_kl_logits(logits, ref_logits):
    """KL(Ber(sigmoid(logits)) || Ber(sigmoid(ref_logits))), elementwise, stable for large |logits|."""
    p = torch.sigmoid(logits)
    return (p * (F.logsigmoid(logits) - F.logsigmoid(ref_logits))
            + (1 - p) * (F.logsigmoid(-logits) - F.logsigmoid(-ref_logits)))


def cam_prior_node_loss(logits, prior_logits, target, weight, graph_idx, tau, lambd, num_d,
                        mask=None, scale=1.0):
    """CAM's node loss (solvers/cam.py) with -tau*H(p) replaced by tau*KL(Ber(p) || Ber(p_ref)).

    mask (0/1, like logits): only these nodes enter the adjoint and KL terms, and both are
    multiplied by scale (num_nodes/m for m sampled nodes per state), so the loss is an unbiased
    estimate of the full one and each node's minimiser is unchanged."""
    prob = torch.sigmoid(logits)
    kl = bernoulli_kl_logits(logits, prior_logits.detach())
    adj = prob*target*torch.exp(-1*(1-weight))
    if mask is not None:
        adj, kl = adj*mask, kl*mask
    return (-scale*g_sum(adj, graph_idx).mean()
            + tau*scale*g_sum(kl, graph_idx).mean()
            + lambd*((g_sum(prob, graph_idx)-num_d)**2).mean())


class CAMPrior(CAM):
    def __init__(self, args):
        super(CAMPrior, self).__init__(args)
        if self.solution_type != 'node':
            raise NotImplementedError('CAMPrior supports node-based problems only (like RLSA)')
        self.prior_d = args.prior_d
        self.prior_eps = args.prior_eps
        self.prior_fixed = args.prior_fixed
        self.prior_tau_min = args.prior_tau_min
        self.prior_policy = args.prior_policy == 'on'
        self.prior_kl_rlsa = args.prior_kl_ref == 'rlsa'
        self.label_mode = args.label_mode
        if self.label_mode in ('sim', 'lin') and args.problem != 'mis':
            # both rebuild the MIS energy from graph.edge_index / graph.A, not from envs/<problem>.py
            raise NotImplementedError(f'--label_mode {self.label_mode} supports --problem mis only')
        self.label_frac = args.label_frac
        if not 0 < self.label_frac <= 1:
            raise ValueError(f'--label_frac must be in (0, 1], got {self.label_frac}')
        self.label_gpus = [g for g in str(args.label_gpus).replace(' ', '').split(',') if g]
        self.label_timeout = args.label_timeout
        self.label_scale = 1.0
        self._label_pool = None
        self._mask_gen = None
        self._label_iter = 0
        self._label_time = []

        # inner schedule: RLSA's TempScheduler over the num_tp training steps (as in solvers/rlsa.py)
        prior_sched = TempScheduler(args.prior_schedule, args.prior_tau0, args.prior_tau1, 0, self.num_tp, 0)
        self.prior_taus = [prior_sched(k) for k in range(self.num_tp)]
        if not all(np.isfinite(t) and t > 0 for t in self.prior_taus):
            raise ValueError(f'prior temperatures must be positive and finite, got {self.prior_taus}')

        # energy whose gradient defines the prior (default: the training energy itself)
        self.prior_beta = args.beta if args.prior_beta is None else args.prior_beta
        if self.prior_beta == args.beta:
            self.prior_problem = self.problem
        else:
            prior_args = copy.copy(args)
            prior_args.beta = self.prior_beta
            env_cls = getattr(importlib.import_module('envs.' + args.problem), 'Env')
            self.prior_problem = env_cls(prior_args)

        self.test_tau_ratio = None
        if args.do_train:
            if args.prior_tau_ratio is not None:
                raise ValueError('--prior_tau_ratio is a test-time option; in training the outer '
                                 'factor follows tau_train')
            self.prior_tau_ref = float(self.temp_scheduler(0))  # tau_train(0)
            if not self.prior_fixed and not self.prior_tau_ref > 0:
                raise ValueError('the outer coupling needs tau_train(0) > 0; use --prior_fixed otherwise')
            if args.checkpoint is None and args.residual_init == 'zero':
                head = zero_output_head(self.model)
                self.accelerator.print(f'CAMPrior: zero-initialised output head "{head}" (sampler = RLSA at init)')
        else:
            self.prior_tau_ref = None
            self.test_tau_ratio = self._load_test_tau_ratio(args)

    # ------------------------------------------------------------------ prior
    def prior_tau_ratio(self):
        """Outer factor tau_prior / tau_rlsa = max(tau_train, tau_min) / tau_train(0)."""
        if self.prior_fixed:
            return 1.0
        if self.test_tau_ratio is not None:
            return self.test_tau_ratio
        floor = min(self.prior_tau_min, self.prior_tau_ref)
        return max(self.tau, floor) / self.prior_tau_ref

    def prior_logits(self, graph, solution, k, ratio, grad=None):
        """RLSA flip logits at inner step k, with the prior temperature tau_rlsa(k) * ratio."""
        if grad is None:
            _, grad = self.prior_problem.energy_func(graph, solution, True)
        delta = grad*(2*solution-1)/2
        diff = delta - topd_threshold(delta, graph.batch, graph.num_graphs, self.prior_d)
        if self.prior_eps != 0:
            diff = diff + self.prior_eps
        return diff/(self.prior_taus[k]*ratio)

    def flip_logits(self, graph, solution, time, k, ratio, grad=None):
        """Total flip logits (prior + network residual; residual only with --prior_policy off)
        and the prior logits."""
        prior = self.prior_logits(graph, solution, k, ratio, grad)
        residual = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
        return (prior + residual if self.prior_policy else residual), prior

    # ------------------------------------------------------------------ training
    def train_batch(self, graph, batch):
        cols = [b.T for b in batch]
        solution, time, weight, target, prior = cols[:5]
        mask = cols[5] if len(cols) > 5 else None          # only with --label_mode sim / --label_frac < 1
        residual = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
        logits = prior + residual if self.prior_policy else residual
        ref = prior if self.prior_kl_rlsa else torch.zeros_like(prior)
        return cam_prior_node_loss(logits, ref, target, weight, graph.batch,
                                   self.tau, self.lambd, self.num_d,
                                   mask=mask, scale=self.label_scale if mask is not None else 1.0)

    @torch.no_grad()
    def generate_data(self, graph):
        if self.label_mode == 'cam' and self.label_frac >= 1:
            return self._generate_data_cam(graph)
        return self._generate_data_labels(graph)

    @torch.no_grad()
    def _generate_data_cam(self, graph):
        # Same rollout as CAM.generate_data (node case); the flip logits include the prior, and the
        # prior logits are stored with each state for the KL term.
        solution = torch.randint(0, 2, size=(graph.num_nodes, self.num_kp),
                                 device=self.accelerator.device, dtype=torch.float)
        all_data, all_time, all_prior = [], [], []
        buffer = BatchBuffer()
        clip_frequency = np.ceil(self.num_tp/self.num_clips)
        ratio = self.prior_tau_ratio()

        for t in range(self.num_tp):
            time = torch.ones_like(solution)*t/self.num_tp
            all_time.append(time)
            all_data.append(solution.clone())
            _, grad = self.problem.energy_func(graph, solution, True)
            logits, prior = self.flip_logits(graph, solution, time, t, ratio,
                                             grad if self.prior_problem is self.problem else None)
            all_prior.append(prior)
            flip_prob = torch.sigmoid(logits)

            if ((t+1)%clip_frequency == 0) or (t==self.num_tp-1):
                all_data = torch.stack(all_data, dim=-1)
                all_time = torch.stack(all_time, dim=-1)
                all_prior = torch.stack(all_prior, dim=-1)
                weight = torch.ones_like(all_time)*(t+1)/self.num_tp
                # CAM's adjoint target, unchanged: energy gradient at the current state
                target = grad.unsqueeze(-1)*(2*all_data-1)

                columns = [c.flatten(1).unbind(1) for c in (all_data, all_time, weight, target, all_prior)]
                buffer.add(list(zip(*columns)))
                all_data, all_time, all_prior = [], [], []

            rr = torch.rand_like(solution)
            solution = torch.where(rr<flip_prob, 1-solution, solution)

        return buffer.get_dataset()

    # ------------------------------------------------------------------ simulated labels / subsampling
    def _label_log(self, msg):
        self.accelerator.print(msg)
        save_dir = getattr(self.args, 'save_dir', None)
        if save_dir and self.accelerator.is_local_main_process:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'sim_labels.log'), 'a') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')

    def label_pool(self):
        if self.label_mode != 'sim' or not self.label_gpus:
            return None
        if self._label_pool is None:
            self._label_pool = SimLabelPool(self.label_gpus, fallback=self.accelerator.device,
                                            timeout=self.label_timeout, log=self._label_log)
            atexit.register(self._label_pool.close)
        return self._label_pool

    def _sample_mask(self, n, cols):
        """m nodes per column, uniformly without replacement; own generator, so the rollout's
        random numbers are the same as without subsampling."""
        dev = self.accelerator.device
        if self._mask_gen is None:
            self._mask_gen = torch.Generator(device=dev)
            self._mask_gen.manual_seed(int(self.args.seed)*1000003 + 7919)
        m = max(1, int(round(self.label_frac*n)))
        if m >= n:
            idx = torch.arange(n, device=dev).unsqueeze(1).expand(n, cols)
        else:
            idx = torch.rand(n, cols, generator=self._mask_gen, device=dev).topk(m, dim=0).indices
        return idx, m                                                    # m x cols

    @torch.no_grad()
    def _generate_data_labels(self, graph):
        """The rollout of _generate_data_cam (same random numbers), then per stored state m sampled
        nodes enter the loss (--label_frac) with CAM's target (--label_mode cam) or the simulated
        label (--label_mode sim). Buffer rows are in the same order as in _generate_data_cam, with
        a 0/1 mask column appended."""
        n, Kc, T = graph.num_nodes, self.num_kp, self.num_tp
        solution = torch.randint(0, 2, size=(n, Kc), device=self.accelerator.device, dtype=torch.float)
        states, times, priors, logits_all, clips = [], [], [], [], []
        clip_frequency = np.ceil(self.num_tp/self.num_clips)
        ratio = self.prior_tau_ratio()
        start = 0
        for t in range(T):
            times.append(torch.ones_like(solution)*t/self.num_tp)
            states.append(solution.clone())
            _, grad = self.problem.energy_func(graph, solution, True)
            logits, prior = self.flip_logits(graph, solution, times[-1], t, ratio,
                                             grad if self.prior_problem is self.problem else None)
            priors.append(prior)
            logits_all.append(logits)
            flip_prob = torch.sigmoid(logits)
            if ((t+1)%clip_frequency == 0) or (t==self.num_tp-1):
                clips.append((start, t, grad.clone()))
                start = t + 1
            rr = torch.rand_like(solution)
            solution = torch.where(rr<flip_prob, 1-solution, solution)
        nexts = states[1:] + [solution]                                  # X_{t+1}

        S = torch.stack(states, dim=1)                                   # n x T x Kc (items t-major)
        idx, m = self._sample_mask(n, T*Kc)                              # m x (T*Kc)
        mask = torch.zeros(n, T*Kc, device=S.device).scatter_(0, idx, 1.0).reshape(n, T, Kc)
        self.label_scale = n/m

        if self.label_mode == 'sim':
            if graph.num_graphs != 1:
                raise NotImplementedError('--label_mode sim needs --batch_size 1')
            t0 = time.time()
            Xn = torch.stack(nexts, dim=1).reshape(n, T*Kc)
            flipped = (Xn != S.reshape(n, T*Kc)).float()
            sel = idx.T.contiguous()                                     # C x m
            sign = 1 - 2*torch.gather(flipped, 0, idx).T                 # C x m
            starts = np.repeat(np.arange(1, T + 1), Kc)
            self._label_iter += 1
            base = (int(self.args.seed) + 1)*10**12 + self._label_iter*10**5
            ew = getattr(graph, 'edge_weight', None)
            common = dict(edge_index=graph.edge_index.cpu().numpy(), n=n,
                          edge_weight=None if ew is None else ew.cpu().numpy(),
                          b=graph.b.cpu().numpy(), temps=[self.prior_taus[j]*ratio for j in range(T)],
                          d=self.prior_d, eps=self.prior_eps, beta_prior=self.prior_beta,
                          beta=self.args.beta, graph_key=self._label_iter)
            lab = simulated_labels(self.label_pool(), self.accelerator.device, common,
                                   Xn.to(torch.uint8).cpu().numpy(), sel.cpu().numpy(),
                                   sign.cpu().numpy(), starts, base + np.arange(T*Kc))
            target_all = torch.zeros(n, T*Kc, device=S.device)
            target_all.scatter_(0, idx, torch.as_tensor(lab, device=S.device).T)
            target_all = target_all.reshape(n, T, Kc)
            self._label_time.append(time.time() - t0)
            if len(self._label_time) == 50:
                pool = self._label_pool
                self._label_log(f'[sim-labels] last 50 graphs: {np.mean(self._label_time):.2f}s/graph '
                                f'for labels; pool stats {pool.stats if pool is not None else "in-process"}')
                self._label_time = []

        if self.label_mode == 'lin':
            if graph.num_graphs != 1:
                raise NotImplementedError('--label_mode lin needs --batch_size 1')
            A = graph.A.to_dense() if graph.A.layout != torch.strided else graph.A
            lab = linear_labels(A.float(), graph.b.float(), torch.stack(states + [solution]),
                                torch.stack(logits_all), torch.stack(priors),
                                [self.prior_taus[j]*ratio for j in range(T)], self.prior_beta,
                                self.args.beta, float(self.tau),
                                neighbors=getattr(self.args, 'lin_neighbors', False))  # T x n x Kc
            target_all = lab.permute(1, 0, 2)*mask                                 # n x T x Kc

        buffer = BatchBuffer()
        for (t0_, t1_, grad) in clips:
            sl = slice(t0_, t1_ + 1)
            all_data = torch.stack(states[sl], dim=-1)
            all_time = torch.stack(times[sl], dim=-1)
            all_prior = torch.stack(priors[sl], dim=-1)
            all_mask = mask[:, sl].permute(0, 2, 1)
            weight = torch.ones_like(all_time)*(t1_+1)/self.num_tp
            if self.label_mode in ('sim', 'lin'):
                target = target_all[:, sl].permute(0, 2, 1)
            else:
                target = grad.unsqueeze(-1)*(2*all_data-1)               # CAM's target, as in _generate_data_cam
            columns = [c.flatten(1).unbind(1) for c in (all_data, all_time, weight, target, all_prior, all_mask)]
            buffer.add(list(zip(*columns)))
        return buffer.get_dataset()

    # ------------------------------------------------------------------ inference
    def evaluate_single(self, graph):
        # Same as ULSolver.evaluate_single (node case) with the prior added to the logits.
        graph.A = torch.sparse_coo_tensor(
            graph.edge_index,
            graph.edge_weight,
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        ratio = self.prior_tau_ratio()
        best_energies = []
        for _ in range(self.num_repetition):
            solution = torch.randint(0, 2, size=(graph.num_nodes, self.num_k),
                                     device=self.accelerator.device, dtype=torch.float)
            for t in range(self.num_t):
                k = int(t*self.num_tp/self.num_t)
                time = torch.ones_like(solution)*k/self.num_tp
                logits, _ = self.flip_logits(graph, solution, time, k, ratio)
                flip_prob = torch.sigmoid(logits)
                if t == self.num_t-1:
                    solution = solution*(1-flip_prob) + flip_prob*(1-solution)
                else:
                    rr = torch.rand_like(solution)
                    solution = torch.where(rr<flip_prob, 1-solution, solution)

            if not self.skip_decode:
                best_energy, _ = self.problem.evaluate(graph, solution)
            else:
                best_energy, _ = self.problem.energy_func(graph, solution)
            best_energies.append(best_energy)

        return torch.cat(best_energies).min()

    # ------------------------------------------------------------------ checkpoints
    def prior_meta(self):
        a = self.args
        return {'tau_ratio': self.prior_tau_ratio(), 'tau_train': float(self.tau),
                'tau_ref': self.prior_tau_ref, 'prior_fixed': self.prior_fixed,
                'prior_tau0': a.prior_tau0, 'prior_tau1': a.prior_tau1,
                'prior_schedule': a.prior_schedule, 'prior_d': a.prior_d, 'prior_eps': a.prior_eps,
                'prior_beta': self.prior_beta, 'prior_tau_min': a.prior_tau_min,
                'num_tp': self.num_tp, 'beta': a.beta, 'global_step': self.global_step,
                'prior_policy': a.prior_policy, 'prior_kl_ref': a.prior_kl_ref,
                'label_mode': a.label_mode, 'label_frac': a.label_frac,
                'lin_neighbors': getattr(a, 'lin_neighbors', False)}

    def save_ckpt(self, save_file):
        super(CAMPrior, self).save_ckpt(save_file)
        if self.accelerator.is_local_main_process:
            with open(save_file + PRIOR_META_SUFFIX, 'w') as f:
                json.dump(self.prior_meta(), f, indent=1)

    def _load_test_tau_ratio(self, args):
        ckpt_path = args.checkpoint if args.checkpoint is not None else os.path.join(args.save_dir, 'best.pt')
        meta_path = ckpt_path + PRIOR_META_SUFFIX
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            current = {'prior_fixed': self.prior_fixed, 'prior_tau0': args.prior_tau0,
                       'prior_tau1': args.prior_tau1, 'prior_schedule': args.prior_schedule,
                       'prior_d': args.prior_d, 'prior_eps': args.prior_eps,
                       'prior_beta': self.prior_beta, 'num_tp': self.num_tp, 'beta': args.beta,
                       'prior_policy': args.prior_policy, 'prior_kl_ref': args.prior_kl_ref}
            older = {'prior_policy': 'on', 'prior_kl_ref': 'rlsa'}   # sidecars written before the flags
            diff = {k: (meta.get(k, older.get(k)), v) for k, v in current.items()
                    if meta.get(k, older.get(k)) != v}
            if diff:
                msg = f'CAMPrior settings differ from training ({meta_path}), (trained, now): {diff}'
                if not args.prior_allow_mismatch:
                    raise ValueError(msg + '; pass the training flags again, or --prior_allow_mismatch '
                                     'to evaluate with different settings on purpose')
                self.accelerator.print('Warning: ' + msg)
        if args.prior_tau_ratio is not None:
            return float(args.prior_tau_ratio)
        if self.prior_fixed:
            return None
        if meta is None:
            raise FileNotFoundError(f'{meta_path} not found: pass --prior_tau_ratio (1.0 = the epoch-0 '
                                    'RLSA scale) or --prior_fixed')
        return float(meta['tau_ratio'])
