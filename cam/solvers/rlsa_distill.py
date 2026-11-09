"""Stage 1 of the distillation warm start (local addition, see cam/UPSTREAM.md).

The conventional warm start of a learned sampler from an existing method: first fit the network to
that method, then train with the sampler's own objective, without the method in the policy or in
the objective (as AS, Havens et al. 2025, App. D, and ASBS, Liu et al. 2025, App. D.3, pretrain on
RDKit samples before their own training). Here the existing method is RLSA and the network is
CAM's; stage 2 is plain CAM training started from the distilled weights:
    main.py --method CAM ... --checkpoint <stage-1 save_dir>/best.pt
(--checkpoint loads the weights only; optimiser, learning-rate and temperature schedules start
from epoch 0, see solvers/meta_solver.py).

Teacher: the prior of CAMPrior at epoch 0, i.e. RLSA with --prior_tau0 / --prior_d / --prior_beta
and its inner schedule over the num_tp steps (solvers/cam_prior.py), outer factor fixed at 1.
With the flags of the CAMPrior p1 runs (--prior_tau0 1, d = 20, the training energy) it is the
sampler those runs start from.

Data (per training graph, as CAM.generate_data): num_kp chains of num_tp RLSA steps from uniform
random states; every visited state x_k is stored with CAM's time input k/num_tp and RLSA's flip
logits a(x_k, k).

Loss: forward KL from the teacher's per-node flip distribution to the student's,
    KL(Ber(sigmoid(a)) || Ber(sigmoid(r_theta(x, k/num_tp)))),
summed over the nodes of each graph and averaged over the stored (state, step) columns, like
CAM's node loss. It equals the soft-label cross-entropy minus the teacher's entropy, so it is 0 at
a perfect fit and its gradient in the student logit is sigmoid(r) - sigmoid(a).

Student: CAM's network with CAM's own initialisation; the flip logits are the network output
alone (--prior_policy off), and validation runs CAM's own inference (ULSolver.evaluate_single),
so eval/object is the score of the distilled CAM sampler. Checkpoints are plain CAM state_dicts
(no CAMPrior .prior.json sidecar).

The CAMPrior options that would put RLSA into the student are forced here: --prior_policy off,
--residual_init default, --prior_fixed (teacher scale 1). Run it through distill_rlsa.py.
"""
import copy

import torch

from .cam_prior import CAMPrior, bernoulli_kl_logits
from .meta_solver import NNSolver, ULSolver
from utils.data_utils import BatchBuffer
from utils.graph_ops import g_sum

FORCED = {'prior_policy': 'off', 'residual_init': 'default', 'prior_fixed': True}


def distill_loss(student_logits, teacher_logits, graph_idx):
    """sum over each graph's nodes of KL(Ber(sigmoid(teacher)) || Ber(sigmoid(student))), mean
    over graphs and columns."""
    return g_sum(bernoulli_kl_logits(teacher_logits.detach(), student_logits), graph_idx).mean()


class RLSADistill(CAMPrior):
    def __init__(self, args):
        changed = {k: (getattr(args, k), v) for k, v in FORCED.items() if getattr(args, k) != v}
        args = copy.copy(args)
        for k, v in FORCED.items():
            setattr(args, k, v)
        super(RLSADistill, self).__init__(args)
        self.accelerator.print(
            f'RLSADistill: student = CAM network alone (CAM init), teacher = RLSA prior '
            f'(tau0 {args.prior_tau0}, d {args.prior_d}, beta {self.prior_beta}, scale 1)'
            + (f'; forced (given, used): {changed}' if changed else ''))

    @torch.no_grad()
    def generate_data(self, graph):
        """num_kp RLSA chains of num_tp steps; columns (state, time, teacher logits)."""
        solution = torch.randint(0, 2, size=(graph.num_nodes, self.num_kp),
                                 device=self.accelerator.device, dtype=torch.float)
        ratio = self.prior_tau_ratio()           # 1.0 (--prior_fixed is forced)
        states, times, teacher = [], [], []
        for t in range(self.num_tp):
            states.append(solution.clone())
            times.append(torch.ones_like(solution)*t/self.num_tp)
            logits = self.prior_logits(graph, solution, t, ratio)
            teacher.append(logits)
            rr = torch.rand_like(solution)
            solution = torch.where(rr < torch.sigmoid(logits), 1-solution, solution)
        buffer = BatchBuffer()
        columns = [torch.stack(c, dim=-1).flatten(1).unbind(1) for c in (states, times, teacher)]
        buffer.add(list(zip(*columns)))
        return buffer.get_dataset()

    def train_batch(self, graph, batch):
        solution, time, teacher = (b.T for b in batch)
        logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
        return distill_loss(logits, teacher, graph.batch)

    def evaluate_single(self, graph):
        # CAM's inference: the network alone, last step soft, then greedy decoding
        return ULSolver.evaluate_single(self, graph)

    def save_ckpt(self, save_file):
        # a plain CAM state_dict, to be loaded by main.py --method CAM --checkpoint
        NNSolver.save_ckpt(self, save_file)
