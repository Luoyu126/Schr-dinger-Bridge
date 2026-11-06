"""ASBS training loop (Adjoint Matching + Corrector Matching), plan §3.4.

Structure follows the official code (train.py / train_loop.py / matcher.py):
  * epochs alternate between an AM stage and a CM stage
    (`am_epochs`, `cm_epochs` epochs per stage, `init_stage` goes first);
  * every epoch draws `resample_size` fresh trajectories into that matcher's
    replay buffer (kept at the last `buffer_size` entries, never cleared) and
    then takes `itr_per_epoch` gradient steps on minibatches of the buffer;
  * the very first stage is the IPF initialisation: AM with h = 0, or CM on
    the uncontrolled base process (u = 0);
  * controller and corrector share one Adam optimiser.

The only f-dependent parts are delegated to the base process:
  bridge_sample (X_t | X_0, X_1), lean_adjoint (a_t from a_1),
  cm_target (grad log p^base(X_1 | X_0)) and step (simulator).

Budget unit: one energy-gradient call = grad U at one terminal sample (AM only).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


class Diverged(RuntimeError):
    pass


@dataclass
class ASBSConfig:
    n_train_steps: int = 100       # simulator steps when generating training samples
    resample_size: int = 512       # fresh samples per epoch (= grad calls per AM epoch)
    sample_batch: int = 512        # simulation batch size
    train_batch: int = 512
    itr_per_epoch: int = 100
    buffer_size: int = 10_000
    duplicates: int = 10
    am_epochs: int = 30
    cm_epochs: int = 30
    init_stage: str = "corrector"  # 'corrector' (u=0 first) | 'adjoint' (h=0 first)
    lr: float = 3e-4
    grad_clip: float | None = None  # optional parameter-gradient norm clip
    alpha_max: float = 100.0
    alpha_sqrt_n_scale: bool = True  # alpha_eff = alpha_max * sqrt(n / 100)
    source_std: float = 1.0
    max_stages: int = 20            # one stage = one AM block + one CM block
    max_grad_calls: int | None = None
    eval_every_stages: int = 1
    explode_threshold: float = 1e4  # |x| above this counts as divergence


class Buffer:
    """GPU replay buffer keeping the most recent `size` entries."""

    def __init__(self, size: int):
        self.size = size
        self.data: dict[str, torch.Tensor] = {}

    def add(self, **kv):
        for k, v in kv.items():
            v = v.detach()
            self.data[k] = v if k not in self.data else torch.cat([self.data[k], v])[-self.size:]

    def __len__(self):
        return 0 if not self.data else next(iter(self.data.values())).shape[0]

    def batches(self, batch: int, n_itr: int, duplicates: int):
        L = len(self)
        dev = next(iter(self.data.values())).device
        idx = torch.cat([torch.randperm(L, device=dev) for _ in range(duplicates)])
        need = batch * n_itr
        while idx.numel() < need:
            idx = torch.cat([idx, torch.randperm(L, device=dev)])
        for i in range(n_itr):
            j = idx[(i * batch) % idx.numel():][:batch]
            yield {k: v[j] for k, v in self.data.items()}


class ASBSTrainer:
    def __init__(self, energy, base, controller, corrector, cfg: ASBSConfig,
                 device="cuda", out_dir: str | Path | None = None,
                 eval_fn: Callable | None = None, log=print):
        self.energy, self.base, self.cfg = energy, base, cfg
        self.u, self.h = controller.to(device), corrector.to(device)
        self.device = torch.device(device)
        self.n = energy.dim
        self.opt = torch.optim.Adam([
            {"params": self.u.parameters(), "lr": cfg.lr},
            {"params": self.h.parameters(), "lr": cfg.lr},
        ])
        self.buf_am, self.buf_cm = Buffer(cfg.buffer_size), Buffer(cfg.buffer_size)
        self.alpha_eff = cfg.alpha_max * (math.sqrt(self.n / 100.0) if cfg.alpha_sqrt_n_scale else 1.0)
        self.out_dir = Path(out_dir) if out_dir else None
        self.eval_fn = eval_fn
        self.log = log
        self.history: list[dict] = []
        self.t_start = time.time()
        self._clip_hits = 0
        self._clip_total = 0

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def simulate(self, B: int, steps: int, controlled: bool = True, return_traj: bool = False):
        x0 = torch.randn(B, self.n, device=self.device) * self.cfg.source_std
        ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
        x, traj = x0, [x0]
        self.u.eval()
        for j in range(steps):
            t, dt = ts[j], ts[j + 1] - ts[j]
            v = self.u(torch.full((B, 1), t, device=self.device), x) if controlled else None
            x = self.base.step(x, t, dt, v=v)
            if return_traj:
                traj.append(x)
        self.u.train()
        if return_traj:
            return torch.stack(traj)
        return x0, x

    def sample(self, N: int, steps: int, controlled: bool = True):
        outs = [self.simulate(min(self.cfg.sample_batch, N - i), steps, controlled)[1]
                for i in range(0, N, self.cfg.sample_batch)]
        return torch.cat(outs)

    def _check_finite(self, x, what):
        if not torch.isfinite(x).all() or x.abs().max() > self.cfg.explode_threshold:
            raise Diverged(f"non-finite / exploding {what}")

    # ---------------------------------------------------------------- matchers
    @torch.no_grad()
    def populate_am(self, init: bool):
        cfg = self.cfg
        for i in range(0, cfg.resample_size, cfg.sample_batch):
            B = min(cfg.sample_batch, cfg.resample_size - i)
            x0, x1 = self.simulate(B, cfg.n_train_steps)
            self._check_finite(x1, "AM samples")
            g = self.energy.grad_u(x1)
            norm = torch.linalg.vector_norm(g, dim=-1)
            self._clip_hits += int((norm > self.alpha_eff).sum())
            self._clip_total += B
            g = g * torch.clamp(self.alpha_eff / (norm + 1e-6), max=1.0)[:, None]
            if not init:
                g = g + self.h(torch.ones(B, 1, device=self.device), x1)
            self.buf_am.add(x0=x0, x1=x1, a1=g)

    @torch.no_grad()
    def populate_cm(self, init: bool):
        cfg = self.cfg
        for i in range(0, cfg.resample_size, cfg.sample_batch):
            B = min(cfg.sample_batch, cfg.resample_size - i)
            x0, x1 = self.simulate(B, cfg.n_train_steps, controlled=not init)
            self._check_finite(x1, "CM samples")
            self.buf_cm.add(x0=x0, x1=x1)

    def am_loss(self, d):
        B = d["x0"].shape[0]
        t = torch.rand(B, 1, device=self.device)
        xt = self.base.bridge_sample(t, d["x0"], d["x1"])
        target = -self.base.lean_adjoint(d["a1"], t)
        return ((self.u(t, xt) - target) ** 2).mean()

    def cm_loss(self, d):
        B = d["x0"].shape[0]
        target = self.base.cm_target(d["x0"], d["x1"])
        return ((self.h(torch.ones(B, 1, device=self.device), d["x1"]) - target) ** 2).mean()

    def train_epoch(self, stage: str, init: bool) -> float:
        cfg = self.cfg
        (self.populate_am if stage == "adjoint" else self.populate_cm)(init)
        buf = self.buf_am if stage == "adjoint" else self.buf_cm
        loss_fn = self.am_loss if stage == "adjoint" else self.cm_loss
        tot = 0.0
        for d in buf.batches(cfg.train_batch, cfg.itr_per_epoch, cfg.duplicates):
            self.opt.zero_grad(set_to_none=True)
            loss = loss_fn(d)
            if not torch.isfinite(loss):
                raise Diverged(f"non-finite {stage} loss")
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.u.parameters()) + list(self.h.parameters()), cfg.grad_clip)
            self.opt.step()
            tot += loss.item()
        return tot / cfg.itr_per_epoch

    # ---------------------------------------------------------------- main loop
    def run(self) -> dict:
        cfg = self.cfg
        order = (["corrector", "adjoint"] if cfg.init_stage == "corrector"
                 else ["adjoint", "corrector"])
        n_ep = {"adjoint": cfg.am_epochs, "corrector": cfg.cm_epochs}
        status = "ok"
        stage_idx = 0
        try:
            for stage_idx in range(1, cfg.max_stages + 1):
                losses = {}
                self._clip_hits = self._clip_total = 0
                for block in order:
                    ls = [self.train_epoch(block, init=(stage_idx == 1 and block == order[0]))
                          for _ in range(n_ep[block])]
                    losses[block] = sum(ls) / len(ls)
                rec = {
                    "stage": stage_idx,
                    "grad_calls": self.energy.grad_calls,
                    "wall": time.time() - self.t_start,
                    "am_loss": losses["adjoint"], "cm_loss": losses["corrector"],
                    "clip_rate": self._clip_hits / max(self._clip_total, 1),
                }
                last = (stage_idx == cfg.max_stages or
                        (cfg.max_grad_calls and self.energy.grad_calls >= cfg.max_grad_calls))
                if self.eval_fn and (stage_idx % cfg.eval_every_stages == 0 or last):
                    rec.update(self.eval_fn(self))
                self._record(rec)
                if last:
                    break
        except Diverged as e:
            status = "diverged"
            self._record({"stage": stage_idx, "grad_calls": self.energy.grad_calls,
                          "wall": time.time() - self.t_start, "diverged": str(e)})
        summary = {"status": status, "stages": stage_idx, "grad_calls": self.energy.grad_calls,
                   "wall": time.time() - self.t_start}
        if self.out_dir:
            (self.out_dir / "summary.json").write_text(json.dumps(summary))
            if status == "ok":
                torch.save({"u": self.u.state_dict(), "h": self.h.state_dict(),
                            "cfg": asdict(cfg)}, self.out_dir / "final.pt")
        return summary

    def _record(self, rec):
        self.history.append(rec)
        self.log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
        if self.out_dir:
            with open(self.out_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
