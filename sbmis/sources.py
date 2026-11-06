"""Source distributions mu for X_0 (plan §3.1, U6).

IsoGaussianSource   N(0, s0^2 I); s0 = 0 is the Dirac delta at 0 (Adjoint Sampling)
GaussianSource      N(m, S) with a fixed mean and covariance
energy_prior        the energy-informed Gaussian of plan U6, built on the corner-equivalent
                    PSD matrix M' = M - lambda_min(M) I (see base.shifted_psd):

    mu(x) ∝ N(x; 0, I) exp(-2 gamma E'_quad(x)) = N(m, (2B)^{-1}),
    B = I/2 + gamma M',  m = -gamma B^{-1} c,  gamma = gamma_rel / lambda_max(M').

    With sigma = 1 the stationary law of B1+ at strength rho is ∝ exp(-2 lam E'_quad),
    lam = rho / lambda_max(M'), so gamma_rel = rho puts the same quadratic tilt into mu
    that B1+ puts into the base drift. M' is PSD, so B is positive definite for every gamma.
"""
from __future__ import annotations

import numpy as np
import torch

from sbmis.base.linear_energy_base import shifted_psd


class IsoGaussianSource:
    def __init__(self, n: int, std: float = 1.0):
        self.n, self.std = n, float(std)
        self.info = {"kind": "delta" if self.std == 0 else "gauss", "std": self.std}

    def sample(self, B: int, device) -> torch.Tensor:
        if self.std == 0:
            return torch.zeros(B, self.n, device=device)
        return torch.randn(B, self.n, device=device) * self.std


class GaussianSource:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, info: dict | None = None):
        mean = np.asarray(mean, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        self.n = mean.shape[0]
        self.mean64 = torch.tensor(mean)
        self.cov64 = torch.tensor(cov)
        self.L64 = torch.linalg.cholesky(self.cov64)
        self.info = info or {"kind": "gaussian"}

    def sample(self, B: int, device) -> torch.Tensor:
        z = torch.randn(B, self.n, device=device)
        return self.mean64.to(z) + z @ self.L64.to(z).T


def energy_prior(energy, gamma_rel: float) -> GaussianSource:
    M, c = energy.M64.numpy(), energy.c64.numpy()
    Ms = shifted_psd(M)
    lam, U = np.linalg.eigh(Ms)
    gamma = gamma_rel / lam.max()
    b = 0.5 + gamma * np.clip(lam, 0.0, None)          # eigenvalues of B (> 0)
    cov = (U / (2 * b)) @ U.T                           # (2B)^{-1}
    mean = -gamma * ((U / b) @ (U.T @ c))               # -gamma B^{-1} c
    return GaussianSource(mean, cov, info={"kind": "energy_prior", "gamma_rel": gamma_rel,
                                           "gamma": float(gamma)})


def build_source(kind: str, energy, std: float = 1.0, gamma_rel: float = 4.0):
    if kind == "gauss":
        return IsoGaussianSource(energy.n, std)
    if kind == "delta":
        return IsoGaussianSource(energy.n, 0.0)
    if kind == "energy_prior":
        return energy_prior(energy, gamma_rel)
    raise ValueError(kind)
