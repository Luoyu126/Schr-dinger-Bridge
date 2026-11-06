"""MIS energy with a double-well confinement term (plan §2).

Encoding: x in R^n, p = (x + 1) / 2.

    E_quad(x) = -1^T p + (lambda_p / 2) p^T A p
    grad E_quad = c + M x,   M = (lambda_p / 4) A,   c = -1/2 + (lambda_p / 4) d
    E_tgt(x)  = E_quad(x) + kappa_w * sum_i (x_i^2 - 1)^2
    nu(x)     ∝ exp(-E_tgt(x) / T)

`pmap` selects how p is computed from x inside E_quad (the double well is unchanged):
    lin    : p = (x + 1) / 2                       (original; unbounded p)
    clamp  : p = clip((x + 1) / 2, 0, 1)
    smooth : p = S(clip((x + 1) / 2, 0, 1)),  S(q) = 3q^2 - 2q^3  (C^1)
With an unbounded p an unselected node at x < -1 has p < 0 and "discounts" its selected
neighbours' conflicts, so at low T the target prefers infeasible states; bounding p in
[0, 1] restores "every corner local minimum is an independent set" for lambda_p > 1.
All three agree on the corners x in {-1, 1}^n. M and c (used by the reference process)
always describe the lin quadratic.
Bounding p alone still leaves metastable infeasible states: inside the box the double well
can hold a conflicting selected node near x = 1 (with smooth, every corner is stationary).
`well` = "out" applies the double well only outside the box, kappa_w * (x^2 - 1)^2 * 1[|x| > 1]
(gradient continuous, zero at |x| = 1); with pmap = clamp the energy is then multilinear in
the box and its local minima are exactly the corners of maximal independent sets
(lambda_p > 1), as in DiffUCO's discrete argument.

All samplers in this package work with the potential U = E_tgt / T
(so nu ∝ exp(-U)); `grad_u` is the gradient that counts as one
"energy gradient call" per sample.
"""
from __future__ import annotations

import numpy as np
import torch


PMAPS = ("lin", "clamp", "smooth")
WELLS = ("full", "out")


def well_mask(x: torch.Tensor, well: str):
    return (x.abs() > 1.0).to(x.dtype) if well == "out" else 1.0


def p_and_dp(x: torch.Tensor, pmap: str):
    """p(x) and dp/dx for the three encodings (elementwise)."""
    q = 0.5 * (x + 1.0)
    if pmap == "lin":
        return q, torch.full_like(x, 0.5)
    inside = ((q > 0.0) & (q < 1.0)).to(x.dtype)
    qc = q.clamp(0.0, 1.0)
    if pmap == "clamp":
        return qc, 0.5 * inside
    if pmap == "smooth":
        return qc * qc * (3.0 - 2.0 * qc), 3.0 * qc * (1.0 - qc) * inside
    raise ValueError(pmap)


class MISEnergy:
    def __init__(
        self,
        adj: np.ndarray | torch.Tensor,
        lambda_p: float = 2.0,
        kappa_w: float = 2.0,
        temp: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        pmap: str = "lin",
        well: str = "full",
    ):
        assert pmap in PMAPS, pmap
        assert well in WELLS, well
        self.pmap, self.well = pmap, well
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
        p, _ = p_and_dp(x, self.pmap)
        return -p.sum(-1) + 0.5 * self.lambda_p * (p * (p @ A)).sum(-1)

    def e_well(self, x: torch.Tensor) -> torch.Tensor:
        return self.kappa_w * (well_mask(x, self.well) * (x * x - 1.0) ** 2).sum(-1)

    def e_tgt(self, x: torch.Tensor) -> torch.Tensor:
        return self.e_quad(x) + self.e_well(x)

    def grad_quad(self, x: torch.Tensor) -> torch.Tensor:
        A, M, c = self._mats(x)
        if self.pmap == "lin":
            return c + x @ M
        p, dp = p_and_dp(x, self.pmap)
        return dp * (-1.0 + self.lambda_p * (p @ A))

    def grad_tgt(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad_quad(x) + 4.0 * self.kappa_w * well_mask(x, self.well) * x * (x * x - 1.0)

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
