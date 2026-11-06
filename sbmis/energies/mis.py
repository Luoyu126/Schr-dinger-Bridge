"""MIS energy with a double-well confinement term (plan §2).

Encoding: x in R^n, p = (x + 1) / 2.

    E_quad(x) = -1^T p + (lambda_p / 2) p^T A p
    grad E_quad = c + M x,   M = (lambda_p / 4) A,   c = -1/2 + (lambda_p / 4) d
    E_tgt(x)  = E_quad(x) + kappa_w * sum_i (x_i^2 - 1)^2
    nu(x)     ∝ exp(-E_tgt(x) / T)

All samplers in this package work with the potential U = E_tgt / T
(so nu ∝ exp(-U)); `grad_u` is the gradient that counts as one
"energy gradient call" per sample.
"""
from __future__ import annotations

import numpy as np
import torch


class MISEnergy:
    def __init__(
        self,
        adj: np.ndarray | torch.Tensor,
        lambda_p: float = 2.0,
        kappa_w: float = 2.0,
        temp: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        A = torch.as_tensor(np.asarray(adj), dtype=torch.float64)
        assert A.ndim == 2 and A.shape[0] == A.shape[1]
        assert torch.allclose(A, A.T), "adjacency must be symmetric"
        assert torch.all(torch.diagonal(A) == 0), "no self loops"
        self.n = A.shape[0]
        self.dim = self.n
        self.lambda_p = float(lambda_p)
        self.kappa_w = float(kappa_w)
        self.temp = float(temp)
        self.device = torch.device(device)
        self.dtype = dtype

        d = A.sum(1)
        # float64 copies (used by the base process eigendecomposition)
        self.A64 = A
        self.M64 = (self.lambda_p / 4.0) * A
        self.c64 = -0.5 + (self.lambda_p / 4.0) * d

        self.A = A.to(device=self.device, dtype=dtype)
        self.deg = d.to(device=self.device, dtype=dtype)
        self.M = self.M64.to(device=self.device, dtype=dtype)
        self.c = self.c64.to(device=self.device, dtype=dtype)
        self.n_edges = int(A.sum().item() // 2)

        self.grad_calls = 0  # cumulative number of per-sample grad_u calls

    # --- energies -------------------------------------------------------
    def _mats(self, x):
        if x.dtype == self.dtype and x.device == self.device:
            return self.A, self.M, self.c
        return (self.A.to(x), self.M.to(x), self.c.to(x))

    def e_quad(self, x: torch.Tensor) -> torch.Tensor:
        A, _, _ = self._mats(x)
        p = 0.5 * (x + 1.0)
        return -p.sum(-1) + 0.5 * self.lambda_p * (p * (p @ A)).sum(-1)

    def e_well(self, x: torch.Tensor) -> torch.Tensor:
        return self.kappa_w * ((x * x - 1.0) ** 2).sum(-1)

    def e_tgt(self, x: torch.Tensor) -> torch.Tensor:
        return self.e_quad(x) + self.e_well(x)

    def grad_quad(self, x: torch.Tensor) -> torch.Tensor:
        _, M, c = self._mats(x)
        return c + x @ M

    def grad_tgt(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad_quad(x) + 4.0 * self.kappa_w * x * (x * x - 1.0)

    # --- potential U = E_tgt / T (sampler interface) ---------------------
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.e_tgt(x) / self.temp

    def grad_u(self, x: torch.Tensor, count: bool = True) -> torch.Tensor:
        if count:
            self.grad_calls += x.shape[0]
        return self.grad_tgt(x) / self.temp


class GaussianEnergy:
    """Toy target nu = N(mean, std^2 I) (used by test T6)."""

    def __init__(self, dim: int, mean: float = 2.0, std: float = 0.5,
                 device="cpu", dtype=torch.float32):
        self.dim = self.n = dim
        self.mean = float(mean)
        self.std = float(std)
        self.device = torch.device(device)
        self.dtype = dtype
        self.grad_calls = 0

    def potential(self, x):
        return 0.5 * ((x - self.mean) ** 2).sum(-1) / self.std**2

    def grad_u(self, x, count: bool = True):
        if count:
            self.grad_calls += x.shape[0]
        return (x - self.mean) / self.std**2
