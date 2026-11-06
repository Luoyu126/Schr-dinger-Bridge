"""Amortized ASBS on a distribution of graphs (plan U7).

Layout: a batch of S samples, sample s living on graph gids[s], is stored padded as
x (S, n_max) with a node mask; flat node index s * n_max + i. Pad nodes are isolated (no
edges), carry x = 0 and are excluded from every loss and metric. The network sees the
disjoint union of the sample graphs.

Per-graph pieces:
  * energy:   E_quad, E_well, grad via index_add over the union's edges (no dense A);
  * base:     the reference process of the single-instance code, per graph. Bases without
              graph dependence (zero, vp) are used as is with n = n_max; graph-dependent ones
              (linear_energy, linear_energy_shift, linear_energy_shift_rot, iso_ou) become a
              BatchedLinearBase holding per-sample theta, b (S, n_max) and eigenvectors U
              (padded with an identity block), so every formula is the tested
              LinearEnergyBase code with batched coordinate changes;
  * source:   gauss / delta / energy_prior per graph.
Per-graph eigendecompositions (float64) are cached in an LRU cache and recomputed on a miss.

The trainer mirrors sbmis.asbs.ASBSTrainer (AM/CM alternation, replay buffers, IPF init,
alpha_max clipping scaled by sqrt(n_g / 100) per sample, Adam shared by u and h, evaluations on
a forked RNG). Each epoch draws `graphs_per_epoch` training graphs and `samples_per_graph`
trajectories on each. Budget unit: one energy-gradient call = one terminal sample.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch

from sbmis.asbs import Diverged
from sbmis.base.linear_energy_base import (GeometricVEBase, LinearEnergyBase, VPBase,
                                           iso_theta_for_var, ou_var, phi1, random_rotation)
from sbmis.datasets import GraphSet

GRAPH_KINDS = ("linear_energy", "linear_energy_shift", "linear_energy_shift_rot", "iso_ou")


# --------------------------------------------------------------------------- batched base
class BatchedLinearBase(LinearEnergyBase):
    """LinearEnergyBase with per-sample theta, b (S, n) and eigenvectors U.

    U is (S, n, n), (G, n, n) with samples grouped as G blocks of S / G consecutive samples,
    or None (no rotation, e.g. the isotropic OU). Only the coordinate changes differ from
    the parent; all closed forms are inherited.
    """

    def __init__(self, theta, b, U, sigma, device, dtype=torch.float32):  # noqa: D401
        self.n = theta.shape[1]
        self.sigma = float(sigma)
        self.device, self.dtype = torch.device(device), dtype
        self.identity = False
        self.theta, self.b, self.U = theta.to(device, dtype), b.to(device, dtype), U
        self.theta64, self.b64 = theta.double(), b.double()
        self._e1 = torch.exp(-self.theta)
        self._phi1_1 = phi1(self.theta, 1.0)
        self._v1 = ou_var(self.theta, 1.0, self.sigma)
        self.rho = self.lam = None

    def _rot(self, x, U):
        if U is None:
            return x
        U = U.to(x)
        if U.shape[0] == x.shape[0]:
            return torch.bmm(x.unsqueeze(1), U).squeeze(1)
        G = U.shape[0]
        return torch.bmm(x.reshape(G, -1, x.shape[1]), U).reshape(x.shape)

    def to_eig(self, x):
        return self._rot(x, self.U)

    def from_eig(self, xt):
        return self._rot(xt, None if self.U is None else self.U.transpose(1, 2))

    def drift(self, t, x):
        raise NotImplementedError("use step / closed forms")


# --------------------------------------------------------------------------- graph pool
@dataclass
class ProblemSpec:
    lambda_p: float = 2.0
    kappa_w: float = 2.0
    temp: float = 0.5
    base: str = "zero"
    rho: float = 0.0
    sigma: float = 1.0
    rot_seed: int = 0
    iso_match_rho: float | None = None
    iso_match: str = "var"
    sigma_min: float = 0.001
    vp_beta0: float = 20.0
    vp_beta1: float = 0.1
    source: str = "gauss"
    source_std: float = 1.0
    prior_gamma: float = 4.0


class GraphPool:
    """A GraphSet with batched energies, reference processes and sources."""

    def __init__(self, gset: GraphSet, spec: ProblemSpec, device="cuda", cache_size=3072,
                 n_max: int | None = None):
        self.g, self.spec, self.device = gset, spec, torch.device(device)
        self.n_max = int(n_max or gset.n.max())
        self.cache: OrderedDict = OrderedDict()   # gid -> params (incl. its bank slot)
        self.cache_size = cache_size
        self.misses = 0
        self._bank = None                          # padded theta, b, U per slot, on device
        self._free_slots = list(range(cache_size))
        sp = spec
        if sp.base == "zero":
            self.static_base = LinearEnergyBase(np.zeros((self.n_max, self.n_max)),
                                                np.zeros(self.n_max), sigma=sp.sigma, rho=0.0,
                                                device=device)
        elif sp.base == "vp":
            self.static_base = VPBase(self.n_max, sp.vp_beta0, sp.vp_beta1, sp.sigma, device=device)
        elif sp.base == "ve_geometric":
            self.static_base = GeometricVEBase(self.n_max, sp.sigma_min, sp.sigma, device=device)
        elif sp.base in GRAPH_KINDS:
            self.static_base = None
        else:
            raise ValueError(sp.base)
        if sp.iso_match_rho is not None:
            assert sp.base == "iso_ou" and sp.source == "gauss"

    # ---- graph structure -------------------------------------------------
    def batch(self, gids) -> SimpleNamespace:
        """Disjoint union of the sample graphs in padded layout."""
        gids = np.asarray(gids)
        S, nm = len(gids), self.n_max
        ptr, E = self.g.edge_ptr, self.g.edges
        cnt = ptr[gids + 1] - ptr[gids]
        sel = np.concatenate([np.arange(ptr[g], ptr[g + 1]) for g in gids]) if S else np.zeros(0, int)
        off = np.repeat(np.arange(S) * nm, cnt)
        e = E[sel].astype(np.int64) + off[:, None]
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        n = self.g.n[gids].astype(np.int64)
        mask = np.arange(nm)[None, :] < n[:, None]
        dev = self.device
        return SimpleNamespace(
            gids=gids, S=S, n=torch.as_tensor(n, device=dev), mask=torch.as_tensor(mask, device=dev),
            src=torch.as_tensor(src, device=dev), dst=torch.as_tensor(dst, device=dev),
            node2sample=torch.arange(S, device=dev).repeat_interleave(nm))

    # ---- energy (padded) ---------------------------------------------------
    def _Ax(self, x, b):
        xf = x.reshape(-1)
        return torch.zeros_like(xf).index_add_(0, b.dst, xf[b.src]).reshape(x.shape)

    def degree(self, b):
        return self._Ax(b.mask.to(torch.float32), b)

    def e_tgt(self, x, b):
        sp, m = self.spec, b.mask.to(x.dtype)
        p = 0.5 * (x + 1.0) * m
        quad = -p.sum(-1) + 0.5 * sp.lambda_p * (p * self._Ax(p, b)).sum(-1)
        well = sp.kappa_w * (((x * x - 1.0) ** 2) * m).sum(-1)
        return quad + well

    def grad_u(self, x, b):
        sp, m = self.spec, b.mask.to(x.dtype)
        c = (-0.5 + 0.25 * sp.lambda_p * self.degree(b)) * m
        g = c + 0.25 * sp.lambda_p * self._Ax(x * m, b) + 4.0 * sp.kappa_w * x * (x * x - 1.0)
        return g * m / sp.temp

    # ---- per-graph reference-process parameters -----------------------------
    def _graph_params(self, gi: int) -> dict:
        if gi in self.cache:
            self.cache.move_to_end(gi)
            return self.cache[gi]
        self.misses += 1
        sp = self.spec
        A = torch.as_tensor(self.g.adj(gi), dtype=torch.float64)
        n = A.shape[0]
        d = A.sum(1)
        M = 0.25 * sp.lambda_p * A
        c = -0.5 + 0.25 * sp.lambda_p * d
        lam_M, U = torch.linalg.eigh(M)
        out = {"n": n}
        needs_shift = sp.base in ("linear_energy_shift", "linear_energy_shift_rot", "iso_ou") \
            or sp.source == "energy_prior"
        lam_s = lam_M - lam_M.min()  # eigenvalues of M - lambda_min I (same eigenvectors)
        if sp.base == "linear_energy":
            lam = 0.0 if sp.rho == 0 else sp.rho / lam_M.abs().max()
            out.update(theta=lam * lam_M, b=lam * (U.T @ c), U=U.float().to(self.device))
        elif sp.base in ("linear_energy_shift", "linear_energy_shift_rot"):
            lam = sp.rho / lam_s.max()
            Ur = U
            if sp.base == "linear_energy_shift_rot":  # B1+ conjugated by a Haar R: U -> R U
                R = torch.as_tensor(random_rotation(n, 1_000_003 * sp.rot_seed + gi))
                Ur = R @ U
            # b = lam U^T c is unchanged by the conjugation (R^T R = I)
            out.update(theta=lam * lam_s, b=lam * (U.T @ c), U=Ur.float().to(self.device))
        elif sp.base == "iso_ou":
            if sp.iso_match_rho is not None:
                lamR = sp.iso_match_rho / lam_s.max()
                th, bb = lamR * lam_s, lamR * (U.T @ c)
                s0 = sp.source_std
                target = float((torch.exp(-2 * th) * s0**2 + ou_var(th, 1.0, sp.sigma)).mean())
                if sp.iso_match == "m2":
                    target += float(((bb * phi1(th, 1.0)) ** 2).mean())
                theta = iso_theta_for_var(target, s0, sp.sigma)
            else:
                theta = sp.rho
            out.update(theta=torch.full((n,), float(theta), dtype=torch.float64),
                       b=torch.zeros(n, dtype=torch.float64), U=None, iso_theta=float(theta))
        if sp.source == "energy_prior":
            gamma = sp.prior_gamma / lam_s.max()
            bev = 0.5 + gamma * lam_s.clamp_min(0)
            out.update(prior_U=U.float().to(self.device),
                       prior_std=(1.0 / (2 * bev)).sqrt().float().to(self.device),
                       prior_mean=(-gamma * (U @ ((U.T @ c) / bev))).float().to(self.device))
        if len(self.cache) >= self.cache_size:
            _, old = self.cache.popitem(last=False)
            if "slot" in old:  # only graph-dependent bases own a bank slot
                self._free_slots.append(old["slot"])
        if "theta" in out:
            self._write_bank(out)
        self.cache[gi] = out
        return out

    def _write_bank(self, p):
        nm, dev = self.n_max, self.device
        if self._bank is None:
            self._bank = {"theta": torch.zeros(self.cache_size, nm, device=dev),
                          "b": torch.zeros(self.cache_size, nm, device=dev),
                          "U": (torch.eye(nm, device=dev).repeat(self.cache_size, 1, 1)
                                if p["U"] is not None else None)}
        k, n = self._free_slots.pop(), p["n"]
        p["slot"] = k
        bk = self._bank
        bk["theta"][k].zero_()
        bk["b"][k].zero_()
        bk["theta"][k, :n] = p["theta"].to(dev, torch.float32)
        bk["b"][k, :n] = p["b"].to(dev, torch.float32)
        if bk["U"] is not None:
            bk["U"][k] = torch.eye(nm, device=dev)
            bk["U"][k, :n, :n] = p["U"]
        p["U"] = None  # the bank holds the only device copy

    def base_ctx(self, gids, grouped: int = 1):
        """Reference process for samples on gids. grouped = K means gids lists each graph once
        and every graph carries K consecutive samples."""
        if self.static_base is not None:
            return self.static_base
        dev = self.device
        assert len(set(int(g) for g in gids)) <= self.cache_size, "cache smaller than batch"
        slots = torch.as_tensor([self._graph_params(int(g))["slot"] for g in gids], device=dev)
        bk = self._bank
        th, bb = bk["theta"][slots], bk["b"][slots]
        U = None if bk["U"] is None else bk["U"][slots]
        if grouped > 1:
            th = th.repeat_interleave(grouped, 0)
            bb = bb.repeat_interleave(grouped, 0)
        return BatchedLinearBase(th, bb, U, self.spec.sigma, dev)

    # ---- sources ---------------------------------------------------------------
    def sample_source(self, gids, grouped: int, mask):
        sp, nm, dev = self.spec, self.n_max, self.device
        S = len(gids) * grouped
        if sp.source == "delta":
            return torch.zeros(S, nm, device=dev)
        z = torch.randn(S, nm, device=dev)
        if sp.source == "gauss":
            return z * sp.source_std * mask
        x = torch.zeros(S, nm, device=dev)
        for k, g in enumerate(gids):
            p = self._graph_params(int(g))
            n = p["n"]
            zz = z[k * grouped:(k + 1) * grouped, :n] * p["prior_std"].to(dev)
            x[k * grouped:(k + 1) * grouped, :n] = p["prior_mean"].to(dev) + zz @ p["prior_U"].to(dev).T
        return x


# --------------------------------------------------------------------------- trainer
@dataclass
class AmortizedConfig:
    graphs_per_epoch: int = 32
    samples_per_graph: int = 8
    n_train_steps: int = 25
    train_batch: int = 64
    itr_per_epoch: int = 50
    buffer_size: int = 10_000
    duplicates: int = 10
    am_epochs: int = 30
    cm_epochs: int = 30
    init_stage: str = "corrector"
    lr: float = 3e-4
    grad_clip: float | None = None
    alpha_max: float = 100.0
    max_stages: int = 20
    max_grad_calls: int | None = None
    eval_every_stages: int = 1
    eval_at_start: bool = True
    explode_threshold: float = 1e4
    fixed_corrector: bool = False
    amp: bool = True
    save_at: tuple = ()
    kl_graphs: int = 16
    seed: int = 0
    sim_chunk: int = 64             # samples per network call during simulation (memory)


class GraphBuffer:
    def __init__(self, size):
        self.size, self.data = size, {}

    def add(self, **kv):
        for k, v in kv.items():
            v = v.detach()
            self.data[k] = v if k not in self.data else torch.cat([self.data[k], v])[-self.size:]

    def __len__(self):
        return 0 if not self.data else next(iter(self.data.values())).shape[0]

    def batches(self, batch, n_itr, duplicates):
        L = len(self)
        dev = next(iter(self.data.values())).device
        idx = torch.cat([torch.randperm(L, device=dev) for _ in range(duplicates)])
        while idx.numel() < batch * n_itr:
            idx = torch.cat([idx, torch.randperm(L, device=dev)])
        for i in range(n_itr):
            j = idx[(i * batch) % idx.numel():][:batch]
            yield {k: v[j] for k, v in self.data.items()}


class AnalyticCorrectorAmortized(torch.nn.Module):
    """h(x1) = grad log p_1^base(x1) for the Dirac source (Adjoint Sampling)."""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, t, x, b):
        base = self.pool.base_ctx(b.gids)
        x = x.reshape(b.S, -1)
        return (base.cm_target(torch.zeros_like(x), x) * b.mask).reshape(-1)


class AmortizedTrainer:
    def __init__(self, pool: GraphPool, u, h, cfg: AmortizedConfig, device="cuda",
                 out_dir=None, eval_fn: Callable | None = None, log=print):
        self.pool, self.cfg, self.device = pool, cfg, torch.device(device)
        self.u, self.h = u.to(device), h.to(device)
        groups = [{"params": list(m.parameters()), "lr": cfg.lr} for m in (self.u, self.h)]
        self.opt = torch.optim.Adam([g for g in groups if g["params"]])
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and self.device.type == "cuda")
        self.buf_am, self.buf_cm = GraphBuffer(cfg.buffer_size), GraphBuffer(cfg.buffer_size)
        self.rng = np.random.default_rng(cfg.seed)
        self.out_dir = Path(out_dir) if out_dir else None
        self.eval_fn, self.log = eval_fn, log
        self.grad_calls, self.t_start, self._epoch = 0, time.time(), 0
        self._saved: set = set()
        self._clip_hits = self._clip_total = 0
        self._ke_sum, self._ke_n = 0.0, 0

    # ---- network evaluation -------------------------------------------------
    def _net(self, net, t, x, b):
        """t (S,), x (S, n_max) -> (S, n_max) masked, in float32."""
        with torch.autocast(self.device.type, dtype=torch.float16,
                            enabled=self.cfg.amp and self.device.type == "cuda"):
            out = net(t, x.reshape(-1), b)
        return out.float().reshape(b.S, -1) * b.mask

    @torch.no_grad()
    def _chunked_net(self, net, t, x, b):
        """_net over whole-graph chunks of about sim_chunk samples (bounded memory)."""
        K = self.cfg.samples_per_graph
        per = max(1, self.cfg.sim_chunk // K) * K
        if b.S <= per:
            return self._net(net, t, x, b)
        outs = [self._net(net, t[i:i + per], x[i:i + per], self.pool.batch(b.gids[i:i + per]))
                for i in range(0, b.S, per)]
        return torch.cat(outs)

    # ---- simulation -------------------------------------------------------------
    @torch.no_grad()
    def simulate(self, gids, K, steps, controlled=True):
        """K trajectories on each graph of gids, simulated in chunks of whole graphs."""
        gids = np.asarray(gids)
        per = max(1, self.cfg.sim_chunk // K)
        parts = [self._simulate_chunk(gids[i:i + per], K, steps, controlled)
                 for i in range(0, len(gids), per)]
        if len(parts) == 1:
            return parts[0]
        b = self.pool.batch(np.repeat(gids, K))
        return (b,) + tuple(torch.cat([p[k] for p in parts]) for k in (1, 2, 3))

    def _simulate_chunk(self, gids, K, steps, controlled):
        b = self.pool.batch(np.repeat(gids, K))
        base = self.pool.base_ctx(gids, grouped=K)
        x0 = self.pool.sample_source(gids, K, b.mask.float())
        x = x0
        ke = torch.zeros(b.S, device=self.device)
        ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
        self.u.eval()
        for j in range(steps):
            t, dt = ts[j], ts[j + 1] - ts[j]
            v = self._net(self.u, torch.full((b.S,), t, device=self.device), x, b) if controlled else None
            if v is not None:
                ke += 0.5 * base.g2_step(t, dt) * (v * v).sum(-1)
            x = base.step(x, t, dt, v=v) * b.mask
        self.u.train()
        return b, x0, x, ke

    def _check(self, x, what):
        if not torch.isfinite(x).all() or x.abs().max() > self.cfg.explode_threshold:
            raise Diverged(f"non-finite / exploding {what}")

    def _draw_graphs(self):
        return self.rng.integers(0, len(self.pool.g), self.cfg.graphs_per_epoch)

    @torch.no_grad()
    def populate_am(self, init):
        cfg = self.cfg
        gids = self._draw_graphs()
        b, x0, x1, ke = self.simulate(gids, cfg.samples_per_graph, cfg.n_train_steps)
        self._check(x1, "AM samples")
        self._ke_sum += float(ke.sum())
        self._ke_n += b.S
        g = self.pool.grad_u(x1, b)
        self.grad_calls += b.S
        alpha = 100.0 if cfg.alpha_max is None else cfg.alpha_max
        alpha = alpha * torch.sqrt(b.n.float() / 100.0)
        norm = torch.linalg.vector_norm(g, dim=-1)
        self._clip_hits += int((norm > alpha).sum())
        self._clip_total += b.S
        g = g * torch.clamp(alpha / (norm + 1e-6), max=1.0)[:, None]
        if not init or cfg.fixed_corrector:
            g = g + self._chunked_net(self.h, torch.ones(b.S, device=self.device), x1, b)
        self.buf_am.add(gid=torch.as_tensor(b.gids, device=self.device), x0=x0, x1=x1, a1=g)

    @torch.no_grad()
    def populate_cm(self, init):
        cfg = self.cfg
        gids = self._draw_graphs()
        b, x0, x1, _ = self.simulate(gids, cfg.samples_per_graph, cfg.n_train_steps,
                                     controlled=not init)
        self._check(x1, "CM samples")
        self.buf_cm.add(gid=torch.as_tensor(b.gids, device=self.device), x0=x0, x1=x1)

    def _loss(self, stage, d):
        gids = d["gid"].cpu().numpy()
        b = self.pool.batch(gids)
        base = self.pool.base_ctx(gids)
        m = b.mask.float()
        if stage == "adjoint":
            t = torch.rand(b.S, 1, device=self.device)
            xt = base.bridge_sample(t, d["x0"], d["x1"]) * m
            target = -base.lean_adjoint(d["a1"], t) * m
            pred = self._net(self.u, t.squeeze(1), xt, b)
        else:
            target = base.cm_target(d["x0"], d["x1"]) * m
            pred = self._net(self.h, torch.ones(b.S, device=self.device), d["x1"], b)
        return (((pred - target) ** 2) * m).sum() / m.sum()

    def train_epoch(self, stage, init):
        cfg = self.cfg
        (self.populate_am if stage == "adjoint" else self.populate_cm)(init)
        buf = self.buf_am if stage == "adjoint" else self.buf_cm
        tot = mx = 0.0
        params = list(self.u.parameters()) + list(self.h.parameters())
        for d in buf.batches(cfg.train_batch, cfg.itr_per_epoch, cfg.duplicates):
            self.opt.zero_grad(set_to_none=True)
            loss = self._loss(stage, d)
            if not torch.isfinite(loss):
                raise Diverged(f"non-finite {stage} loss")
            self.scaler.scale(loss).backward()
            if cfg.grad_clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            tot += loss.item()
            mx = max(mx, loss.item())
        self._epoch += 1
        if self.out_dir:
            with open(self.out_dir / "epochs.jsonl", "a") as f:
                f.write(json.dumps({"epoch": self._epoch, "stage": stage,
                                    "loss": tot / cfg.itr_per_epoch, "loss_max": mx,
                                    "grad_calls": self.grad_calls,
                                    "wall": time.time() - self.t_start}) + "\n")
        return tot / cfg.itr_per_epoch

    # ---- evaluation ----------------------------------------------------------------
    @torch.no_grad()
    def sample_graphs(self, gids, K, steps):
        """K samples on each graph; returns padded x (len(gids) * K, n_max) and the batch."""
        b, _, x, _ = self.simulate(np.asarray(gids), K, steps)
        return b, x

    def kl_to_ref(self):
        gids = self._draw_graphs()[: self.cfg.kl_graphs]
        _, _, _, ke = self.simulate(gids, self.cfg.samples_per_graph, self.cfg.n_train_steps)
        return float(ke.mean())

    def _evaluate(self):
        devs = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            state = self.rng.bit_generator.state
            res = dict(self.eval_fn(self)) if self.eval_fn else {}
            res["kl_ref"] = self.kl_to_ref()
            self.rng.bit_generator.state = state
        return res

    # ---- main loop ---------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        order = ["corrector", "adjoint"] if cfg.init_stage == "corrector" else ["adjoint", "corrector"]
        if cfg.fixed_corrector:
            order = ["adjoint"]
        n_ep = {"adjoint": cfg.am_epochs, "corrector": cfg.cm_epochs}
        status, stage_idx = "ok", 0
        if cfg.eval_at_start and self.eval_fn:
            self._record({"stage": 0, "grad_calls": 0, "wall": 0.0, **self._evaluate()})
        try:
            for stage_idx in range(1, cfg.max_stages + 1):
                losses = {}
                self._clip_hits = self._clip_total = 0
                self._ke_sum, self._ke_n = 0.0, 0
                for blk in order:
                    ls = [self.train_epoch(blk, init=(stage_idx == 1 and blk == order[0]))
                          for _ in range(n_ep[blk])]
                    losses[blk] = sum(ls) / len(ls)
                rec = {"stage": stage_idx, "grad_calls": self.grad_calls,
                       "wall": time.time() - self.t_start,
                       "am_loss": losses["adjoint"], "cm_loss": losses.get("corrector"),
                       "clip_rate": self._clip_hits / max(self._clip_total, 1),
                       "ctrl_energy_stage": self._ke_sum / max(self._ke_n, 1),
                       "cache_misses": self.pool.misses}
                last = stage_idx == cfg.max_stages or (
                    cfg.max_grad_calls and self.grad_calls >= cfg.max_grad_calls)
                if self.eval_fn and (stage_idx % cfg.eval_every_stages == 0 or last):
                    rec.update(self._evaluate())
                self._record(rec)
                self._maybe_save()
                if last:
                    break
        except Diverged as e:
            status = "diverged"
            self._record({"stage": stage_idx, "grad_calls": self.grad_calls,
                          "wall": time.time() - self.t_start, "diverged": str(e)})
        summary = {"status": status, "stages": stage_idx, "grad_calls": self.grad_calls,
                   "wall": time.time() - self.t_start}
        if self.out_dir:
            (self.out_dir / "summary.json").write_text(json.dumps(summary))
            if status == "ok":
                torch.save({"u": self.u.state_dict(), "h": self.h.state_dict(),
                            "cfg": asdict(cfg)}, self.out_dir / "final.pt")
        return summary

    def _maybe_save(self):
        if not self.out_dir:
            return
        due = [m for m in self.cfg.save_at if m not in self._saved and self.grad_calls >= m]
        if due:
            torch.save({"u": self.u.state_dict(), "h": self.h.state_dict(),
                        "grad_calls": self.grad_calls, "milestones": due,
                        "cfg": asdict(self.cfg)}, self.out_dir / f"ckpt_{self.grad_calls}.pt")
            self._saved.update(due)

    def _record(self, rec):
        self.log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
        if self.out_dir:
            with open(self.out_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- evaluation
def decode_metrics(pool: GraphPool, b, x, K: int, opt: dict | None = None) -> dict:
    """Per-graph decode of K samples per graph (padded x). Returns averages over graphs."""
    from sbmis.evaluate import Decoder
    rows = []
    gids = b.gids[::K]
    for k, g in enumerate(gids):
        n = int(pool.g.n[g])
        xs = x[k * K:(k + 1) * K, :n]
        dec = Decoder(pool.g.adj(int(g)), device=x.device)
        s_raw = dec.hard(xs)
        conf = dec.conflicts(s_raw)
        sol = dec.extend(dec.repair(s_raw))
        size = sol.sum(-1)
        r = {"mis_size_mean": float(size.mean()), "best_of_K": float(size.max()),
             # same definitions as sbmis.evaluate.mis_metrics (groups of k samples)
             **{f"best_of_{k}": float(size[: (K // k) * k].view(-1, k).max(1).values.mean())
                for k in (1, 16) if K >= k},
             "raw_feasible_frac": float((conf == 0).float().mean()),
             "raw_conflict_edge_frac": float((conf / max(dec.n_edges, 1)).mean()),
             "raw_size_mean": float(s_raw.sum(-1).mean()),
             "n_distinct": int(torch.unique(sol, dim=0).shape[0]),
             "n_distinct_64": int(torch.unique(sol[:64], dim=0).shape[0])}
        if opt is not None:
            o = opt[int(g)]["size"]
            r.update(gap_mean=float(1 - size.mean() / o), gap_best=float(1 - size.max() / o),
                     ratio_mean=float(size.mean() / o), ratio_best=float(size.max() / o))
            if K >= 16:
                r["ratio_best_of_16"] = r["best_of_16"] / o
        rows.append(r)
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["n_graphs"] = len(rows)
    return out
