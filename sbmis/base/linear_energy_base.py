"""Base processes for ASBS on MIS (plan §3.2, §3.3, §3.5).

LinearEnergyBase:   dX = -lam (c + M X) dt + sigma dW      (B1; lam = 0 is B0)
                    also used with M -> M - lambda_min(M) I (B1+, see shifted_psd), with a
                    randomly rotated copy of B1+ (control) and with M = I, c = 0 (isotropic OU)
GeometricVEBase:    dX = g(t) dW, geometric g(t)           (B0-S, f = 0, ASBS paper's VE)
VPBase:             dX = -beta_t/2 X dt + sigma sqrt(beta_t) dW   (official VPSDE, control)

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
        # (tau is recomputed in float64: casting a float32 1 - t would break t + tau = 1,
        #  which the cancellation in G amplifies by ~1/theta)
        th64 = th.double().expand(t.shape[0], -1)
        t64 = t.double()
        tau64 = 1.0 - t64
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


def shifted_psd(M) -> np.ndarray:
    """M - lambda_min(M) I.

    On the corners {-1, 1}^n every x_i^2 = 1, so x^T (M + mu I) x = x^T M x + mu n: the
    quadratic energy changes by a constant only and encodes the same MIS problem. With
    mu = -lambda_min(M) the matrix is PSD, so the base drift never expands any direction
    and the lean adjoint exp(-theta (1 - t)) is bounded by 1.
    """
    M = np.asarray(M, dtype=np.float64)
    return M - np.linalg.eigvalsh(M).min() * np.eye(M.shape[0])


def random_rotation(n: int, seed: int) -> np.ndarray:
    """Haar-distributed orthogonal matrix (QR of a Gaussian with the sign fix)."""
    g = np.random.default_rng(seed).standard_normal((n, n))
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))


def terminal_var(base: "LinearEnergyBase", s0: float = 1.0) -> float:
    """Mean per-coordinate variance of X_1 under u = 0 when X_0 ~ N(0, s0^2 I)."""
    th = base.theta64
    var = torch.exp(-2 * th) * s0**2 + ou_var(th, 1.0, base.sigma)
    return float(var.mean())


def iso_theta_for_var(target_var: float, s0: float = 1.0, sigma: float = 1.0) -> float:
    """theta of the isotropic OU f = -theta x whose X_1 variance (from N(0, s0^2)) is target_var.

    The variance is strictly decreasing in theta, from s0^2 + sigma^2 at theta = 0 to 0.
    """
    def var(th):
        th = torch.tensor([th], dtype=torch.float64)
        return float(torch.exp(-2 * th) * s0**2 + ou_var(th, 1.0, sigma))
    assert 0 < target_var < s0**2 + sigma**2, target_var
    lo, hi = 0.0, 1.0
    while var(hi) > target_var:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if var(mid) > target_var else (lo, mid)
    return 0.5 * (lo + hi)


class VPBase:
    """Official VPSDE: dX = -beta_t / 2 X dt + sigma sqrt(beta_t) dW, beta_t = lerp(b0, b1, t).

    Isotropic linear drift without any problem information (control group). With
    X_0 ~ N(0, sigma^2) the marginal stays N(0, sigma^2). Closed forms use
    B(s, t) = int_s^t beta: transition mean exp(-B/2) x_s, variance sigma^2 (1 - exp(-B)).
    The simulator integrates the linear part exactly with v held constant over a step
    (the official code uses Euler-Maruyama; B1 uses the same exact treatment).
    """

    has_drift = True
    identity = False

    def __init__(self, n, beta0=20.0, beta1=0.1, sigma=1.0, device="cpu", dtype=torch.float32):
        self.n = n
        self.beta0, self.beta1, self.sigma = float(beta0), float(beta1), float(sigma)
        self.device, self.dtype = torch.device(device), dtype
        self.rho, self.lam = None, None
        self.info = {"kind": "vp", "beta0": self.beta0, "beta1": self.beta1}

    def _B(self, a, b):
        """int_a^b beta_s ds (works for floats and tensors)."""
        return self.beta0 * (b - a) + 0.5 * (self.beta1 - self.beta0) * (b * b - a * a)

    def beta(self, t):
        return self.beta0 + (self.beta1 - self.beta0) * t

    def drift(self, t, x):
        return -0.5 * self.beta(float(t)) * x

    def diff(self, t):
        t = torch.as_tensor(t, dtype=self.dtype)
        return self.sigma * torch.sqrt(self.beta(t))

    def transition_mean_var(self, xs, s, t):
        Bst = self._B(s, t)
        return math.exp(-0.5 * Bst) * xs, self.sigma**2 * (-math.expm1(-Bst))

    def bridge_mean_var(self, t, x0, x1):
        t64 = t.double()
        B0t, Bt1, B01 = self._B(0.0, t64), self._B(t64, 1.0), self._B(0.0, 1.0)
        den = -math.expm1(-B01)
        w0 = torch.exp(-0.5 * B0t) * (-torch.expm1(-Bt1)) / den
        w1 = torch.exp(-0.5 * Bt1) * (-torch.expm1(-B0t)) / den
        vr = (-torch.expm1(-B0t)) * (-torch.expm1(-Bt1)) / den
        w0, w1, vr = (a.to(x0.dtype) for a in (w0, w1, vr))
        return w0 * x0 + w1 * x1, (self.sigma**2 * vr).clamp_min(0.0).expand_as(x0)

    def bridge_sample(self, t, x0, x1, z=None):
        mean, var = self.bridge_mean_var(t, x0, x1)
        if z is None:
            z = torch.randn_like(x0)
        return mean + var.sqrt() * z

    def lean_adjoint(self, a1, t):
        # -da = a grad f dt, grad f = -beta_t / 2 I  =>  a_t = exp(-B(t, 1) / 2) a_1
        return a1 * torch.exp(-0.5 * self._B(t.to(a1), 1.0))

    def cm_target(self, x0, x1):
        B01 = self._B(0.0, 1.0)
        return (math.exp(-0.5 * B01) * x0 - x1) / (self.sigma**2 * (-math.expm1(-B01)))

    def step(self, x, t, dt, v=None, z=None):
        """Exact for the linear part: x' = e x + 2 sigma^2 (1 - e) v + sigma sqrt(1 - e^2) z,
        e = exp(-B(t, t + dt) / 2); the control drift is g^2 v = sigma^2 beta_t v."""
        if z is None:
            z = torch.randn_like(x)
        e = math.exp(-0.5 * self._B(t, t + dt))
        out = e * x + self.sigma * math.sqrt(1 - e * e) * z
        if v is not None:
            out = out + 2 * self.sigma**2 * (1 - e) * v
        return out


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
    """kind:
    zero                     B0   f = 0, constant sigma
    linear_energy            B1   f = -lam (c + M x), max|theta| = rho
    linear_energy_shift      B1+  same with M - lambda_min(M) I (PSD; same energy on corners)
    linear_energy_shift_rot  control: B1+ conjugated by a Haar rotation R (M -> R M R^T,
                             c -> R c): identical dynamics in rotated coordinates, but the
                             alignment with the hypercube / graph is destroyed
    iso_ou                   control: f = -theta x; theta = rho, or (iso_match_rho=R) the theta
                             whose u = 0 terminal variance equals that of B1+ at rho = R
    ve_geometric             B0-S f = 0, geometric noise (sigma = sigma_max)
    vp                       control: official VPSDE (beta0 -> beta1), sigma = stationary std
    """
    M, c = energy.M64.numpy(), energy.c64.numpy()
    mk = lambda M_, c_, **a: LinearEnergyBase(M_, c_, sigma=sigma, device=device, dtype=dtype, **a)
    if kind == "zero":
        b = mk(M, c, rho=0.0)
    elif kind == "linear_energy":
        b = mk(M, c, rho=rho)
    elif kind == "linear_energy_shift":
        b = mk(shifted_psd(M), c, rho=rho)
        b.info = {"shift_mu": float(-np.linalg.eigvalsh(M).min())}
    elif kind == "linear_energy_shift_rot":
        R = random_rotation(len(c), kw.get("rot_seed", 0))  # x -> R x conjugates B1+
        b = mk(R @ shifted_psd(M) @ R.T, R @ c, rho=rho)
        b.info = {"rot_seed": kw.get("rot_seed", 0)}
    elif kind == "iso_ou":
        R = kw.get("iso_match_rho")
        if R is not None:
            ref = mk(shifted_psd(M), c, rho=R)
            theta = iso_theta_for_var(terminal_var(ref, kw.get("source_std", 1.0)),
                                      kw.get("source_std", 1.0), sigma)
        else:
            theta = rho
        n = len(c)
        b = mk(np.eye(n), np.zeros(n), lam=theta)
        b.info = {"iso_theta": theta, "iso_match_rho": R}
    elif kind == "ve_geometric":
        b = GeometricVEBase(energy.n, kw.get("sigma_min", 0.01), kw.get("sigma_max", sigma),
                            device=device, dtype=dtype)
    elif kind == "vp":
        b = VPBase(energy.n, kw.get("vp_beta0", 20.0), kw.get("vp_beta1", 0.1), sigma,
                   device=device, dtype=dtype)
    else:
        raise ValueError(kind)
    b.info = {"kind": kind, **getattr(b, "info", {})}
    return b
