"""Base processes for ASBS on MIS (plan §3.2, §3.3, §3.5).

LinearEnergyBase:   dX = -lam (c + M X) dt + sigma dW      (B1; lam = 0 is B0)
GeometricVEBase:    dX = g(t) dW, geometric g(t)           (B0-S, f = 0, ASBS paper's VE)

All public methods take / return tensors in the original coordinates x
(shape (B, n)); the linear base works internally in the eigenbasis of M,
x_tilde = U^T x (row-vector form: x_tilde = x @ U). Noise arguments `z` are
standard normals in original coordinates, rotated internally, so that for
rho = 0 (where U = I is used) every method coincides exactly with the f = 0
implementation of the official ASBS code.

Controller convention (same as the official code): the network output v is
used as drift f + g(t)^2 v, and Adjoint Matching regresses v onto -a_t.
"""
from __future__ import annotations

import math

import numpy as np
import torch

SMALL = 1e-4  # |theta| below which Taylor expansions are used (plan §5)
VAR_MIN = 1e-12


def _safe(theta):
    return torch.where(theta.abs() < SMALL, torch.ones_like(theta), theta)


def phi1(theta: torch.Tensor, delta) -> torch.Tensor:
    """(1 - exp(-theta * delta)) / theta, with the theta -> 0 limit."""
    delta = torch.as_tensor(delta, dtype=theta.dtype, device=theta.device)
    small = theta.abs() < SMALL
    ts = _safe(theta)
    exact = -torch.expm1(-ts * delta) / ts
    taylor = delta - theta * delta**2 / 2 + theta**2 * delta**3 / 6
    return torch.where(small, taylor, exact)


def ou_var(theta: torch.Tensor, delta, sigma: float) -> torch.Tensor:
    """sigma^2 (1 - exp(-2 theta delta)) / (2 theta)."""
    return sigma**2 * phi1(2.0 * theta, delta)


class LinearEnergyBase:
    """Closed-form linear OU base process in the eigenbasis of M."""

    has_drift = True

    def __init__(
        self,
        M: np.ndarray | torch.Tensor,
        c: np.ndarray | torch.Tensor,
        sigma: float = 1.0,
        rho: float | None = None,
        lam: float | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        M = torch.as_tensor(np.asarray(M), dtype=torch.float64)
        c = torch.as_tensor(np.asarray(c), dtype=torch.float64)
        n = M.shape[0]
        assert (rho is None) != (lam is None), "give exactly one of rho, lam"
        self.n = n
        self.sigma = float(sigma)
        self.device = torch.device(device)
        self.dtype = dtype

        Lam, U = torch.linalg.eigh(M)
        orth_err = (U.T @ U - torch.eye(n, dtype=torch.float64)).abs().max().item()
        assert orth_err < 1e-8, f"eigenvectors not orthonormal: {orth_err}"
        self.max_abs_eig = Lam.abs().max().item()

        if lam is None:
            lam = 0.0 if rho == 0 else float(rho) / self.max_abs_eig
        self.lam = float(lam)
        self.rho = self.lam * self.max_abs_eig if rho is None else float(rho)
        if self.lam == 0.0:
            # exact f = 0 path (B0): no rotation at all
            U = torch.eye(n, dtype=torch.float64)
            Lam = torch.zeros(n, dtype=torch.float64)
        theta = self.lam * Lam
        b = self.lam * (U.T @ c)  # lam * c_tilde

        self.U64, self.theta64, self.b64 = U, theta, b
        self.M64, self.c64 = M, c
        self.U = U.to(self.device, dtype)
        self.theta = theta.to(self.device, dtype)
        self.b = b.to(self.device, dtype)
        self.identity = self.lam == 0.0

        # constants at delta = 1 (CM target, p_1^base)
        self._e1 = torch.exp(-theta).to(self.device, dtype)
        self._phi1_1 = phi1(theta, 1.0).to(self.device, dtype)
        self._v1 = ou_var(theta, 1.0, self.sigma).to(self.device, dtype)

    # ---- coordinate changes ------------------------------------------------
    def to_eig(self, x):
        return x if self.identity else x @ self.U.to(x)

    def from_eig(self, xt):
        return xt if self.identity else xt @ self.U.to(xt).T

    def _th(self, x):
        return self.theta.to(x), self.b.to(x)

    # ---- the base drift f(x) = -lam (c + M x) -----------------------------
    def drift(self, t, x):
        if self.identity:
            return torch.zeros_like(x)
        M, c = self.M64.to(x), self.c64.to(x)
        return -self.lam * (c + x @ M)

    def diff(self, t):
        return torch.full_like(torch.as_tensor(t, dtype=self.dtype), self.sigma)

    # ---- (1) transition density -----------------------------------------
    def transition_mean_var(self, xs, delta):
        """Mean (orig coords) and eigen-variances of X_{s+delta} | X_s = xs."""
        th, b = self._th(xs)
        xe = self.to_eig(xs)
        mean_e = torch.exp(-th * delta) * xe - b * phi1(th, delta)
        return self.from_eig(mean_e), ou_var(th, delta, self.sigma)

    def transition_sample(self, xs, delta, z=None):
        mean, var = self.transition_mean_var(xs, delta)
        if z is None:
            z = torch.randn_like(xs)
        return mean + self.from_eig(var.clamp_min(VAR_MIN).sqrt() * self.to_eig(z))

    # ---- (2) base bridge p(X_t | X_0, X_1) -------------------------------
    def bridge_mean_var(self, t, x0, x1):
        """t: (B, 1). Returns mean (B, n) in orig coords and eigen-variance (B, n)."""
        th, b = self._th(x0)
        t = t.to(x0)
        tau = 1.0 - t
        x0e, x1e = self.to_eig(x0), self.to_eig(x1)
        if self.identity:
            mean = tau * x0 + t * x1
            var = (self.sigma**2) * (t * tau).expand_as(x0)
            return mean, var
        # work in float64 for the hyperbolic ratios, then cast back
        th64 = th.double().expand(t.shape[0], -1)
        t64, tau64 = t.double(), tau.double()
        small = th64.abs() < SMALL
        ts = _safe(th64)
        S1 = torch.sinh(ts)
        St, Stau = torch.sinh(ts * t64), torch.sinh(ts * tau64)
        w0 = torch.where(small, tau64 * (1 + th64**2 * (tau64**2 - 1) / 6), Stau / S1)
        w1 = torch.where(small, t64 * (1 + th64**2 * (t64**2 - 1) / 6), St / S1)
        G = torch.where(small, th64 * t64 * tau64 / 2, (1 - (Stau + St) / S1) / ts)
        vr = torch.where(
            small,
            t64 * tau64 * (1 + th64**2 * (t64**2 + tau64**2 - 1) / 6),
            St * Stau / (ts * S1),
        )
        w0, w1, G, vr = (a.to(x0.dtype) for a in (w0, w1, G, vr))
        mean_e = w0 * x0e + w1 * x1e - b * G
        return self.from_eig(mean_e), (self.sigma**2) * vr

    def bridge_sample(self, t, x0, x1, z=None):
        mean, var = self.bridge_mean_var(t, x0, x1)
        if z is None:
            z = torch.randn_like(x0)
        std = var.clamp_min(0.0).sqrt()
        return mean + self.from_eig(std * self.to_eig(z))

    # ---- (3) lean adjoint a_t = exp(-lam M (1 - t)) a_1 ----------------------
    def lean_adjoint(self, a1, t):
        if self.identity:
            return a1
        th, _ = self._th(a1)
        return self.from_eig(torch.exp(-th * (1.0 - t.to(a1))) * self.to_eig(a1))

    # ---- CM target: grad_{x1} log p^base(x1 | x0) ---------------------------
    def cm_target(self, x0, x1):
        e1, p1, v1 = (a.to(x0) for a in (self._e1, self._phi1_1, self._v1))
        if self.identity:
            return (x0 - x1) / v1
        _, b = self._th(x0)
        r = self.to_eig(x1) - e1 * self.to_eig(x0) + b * p1
        return -self.from_eig(r / v1)

    # ---- p_1^base when X_0 ~ N(m0, s0^2 I)  (analytic first CM, plan §3.4) ---
    def marginal1_score(self, x1, s0: float = 1.0, m0: float = 0.0):
        e1, p1, v1 = (a.to(x1) for a in (self._e1, self._phi1_1, self._v1))
        _, b = self._th(x1)
        m0e = self.to_eig(torch.full_like(x1[:1], m0))
        mean = e1 * m0e - b * p1
        var = e1**2 * s0**2 + v1
        return -self.from_eig((self.to_eig(x1) - mean) / var)

    # ---- §3.5 exponential integrator ------------------------------------------
    def step(self, x, t, dt, v=None, z=None):
        """One step from t to t + dt. Control drift is sigma^2 * v (v = net output)."""
        if z is None:
            z = torch.randn_like(x)
        sig = self.sigma
        if self.identity:
            out = x + sig * math.sqrt(dt) * z
            if v is not None:
                out = out + (sig**2) * dt * v
            return out
        th, b = self._th(x)
        forcing = -b if v is None else (-b + (sig**2) * self.to_eig(v))
        xe = (torch.exp(-th * dt) * self.to_eig(x) + phi1(th, dt) * forcing
              + ou_var(th, dt, sig).clamp_min(VAR_MIN).sqrt() * self.to_eig(z))
        return self.from_eig(xe)


class GeometricVEBase:
    """f = 0 with the geometric noise schedule of the ASBS paper (group B0-S).

    g(t) = s_min (s_max / s_min)^(1 - t) sqrt(2 log(s_max / s_min)),
    int_0^t g^2 = s_max^2 (1 - (s_max/s_min)^(-2t)).
    Bridge / CM target follow VESDE.sample_posterior / cond_score of the official code.
    """

    has_drift = False
    identity = True

    def __init__(self, n, sigma_min=0.01, sigma_max=1.0, device="cpu", dtype=torch.float32):
        self.n = n
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.ratio = self.sigma_max / self.sigma_min
        self.total_var = self.sigma_max**2 - self.sigma_min**2
        self.device, self.dtype = torch.device(device), dtype
        self.rho, self.lam = 0.0, 0.0

    def _int_g2(self, t):  # official code's _diffsquare_integral
        return (self.sigma_max**2) * (1 - self.ratio ** (-2 * t))

    def diff(self, t):
        t = torch.as_tensor(t, dtype=self.dtype)
        return self.sigma_min * self.ratio ** (1 - t) * math.sqrt(2 * math.log(self.ratio))

    def drift(self, t, x):
        return torch.zeros_like(x)

    def bridge_mean_var(self, t, x0, x1):
        r = self._int_g2(t.to(x0)) / self.total_var
        mean = (1 - r) * x0 + r * x1
        var = (self.total_var * r * (1 - r)).clamp_min(0.0).expand_as(x0)
        return mean, var

    def bridge_sample(self, t, x0, x1, z=None):
        mean, var = self.bridge_mean_var(t, x0, x1)
        if z is None:
            z = torch.randn_like(x0)
        return mean + var.sqrt() * z

    def lean_adjoint(self, a1, t):
        return a1

    def cm_target(self, x0, x1):
        return (x0 - x1) / self._int_g2(torch.ones(1, device=x0.device, dtype=x0.dtype))

    def step(self, x, t, dt, v=None, z=None):
        if z is None:
            z = torch.randn_like(x)
        g = float(self.diff(float(t)))
        out = x + g * math.sqrt(dt) * z
        if v is not None:
            out = out + g**2 * dt * v
        return out


def build_base(kind: str, energy, sigma=1.0, rho=0.0, device="cpu", dtype=torch.float32, **kw):
    if kind == "zero":
        return LinearEnergyBase(energy.M64, energy.c64, sigma=sigma, rho=0.0,
                                device=device, dtype=dtype)
    if kind == "linear_energy":
        return LinearEnergyBase(energy.M64, energy.c64, sigma=sigma, rho=rho,
                                device=device, dtype=dtype)
    if kind == "ve_geometric":
        return GeometricVEBase(energy.n, kw.get("sigma_min", 0.01), kw.get("sigma_max", sigma),
                               device=device, dtype=dtype)
    raise ValueError(kind)
