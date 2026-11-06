"""Reference-process variants of the main experiment: B1+, its rotated control,
isotropic OU (variance matched) and the official VP SDE."""
import itertools

import numpy as np
import pytest
import torch

from asbs_reference import vp_coeff2, vp_cond_score, vp_sample_posterior
from sbmis.base.linear_energy_base import (LinearEnergyBase, VPBase, build_base, random_rotation,
                                           shifted_psd, terminal_var)
from sbmis.energies.mis import MISEnergy
from sbmis.graphs import er_graph

F64 = torch.float64


def _energy(n=10, seed=3):
    return MISEnergy(er_graph(n, 0.3, seed=seed), 2.0, 2.0, 0.5, device="cpu", dtype=F64)


def test_shift_same_energy_on_all_corners():
    en = _energy()
    M, c = en.M64.numpy(), en.c64.numpy()
    Ms = shifted_psd(M)
    assert np.linalg.eigvalsh(Ms).min() > -1e-10
    X = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=len(c))), dtype=F64)
    e = en.e_quad(X)
    quad = lambda MM: X @ torch.tensor(c) + 0.5 * ((X @ torch.tensor(MM)) * X).sum(-1)
    # E_quad = const + c^T x + x^T M x / 2, and the shifted M only adds a constant on corners
    for MM in (M, Ms):
        d = quad(MM) - e
        assert (d - d[0]).abs().max() < 1e-10


@pytest.mark.parametrize("rho", [2.0, 8.0])
def test_shift_base_is_contractive(rho):
    b = build_base("linear_energy_shift", _energy(), rho=rho, dtype=F64)
    assert b.theta64.min() > -1e-10 and abs(b.theta64.max() - rho) < 1e-10
    a1 = torch.randn(64, b.n, dtype=F64)
    for t in (0.0, 0.5, 1.0):
        at = b.lean_adjoint(a1, torch.full((64, 1), t, dtype=F64))
        assert (at.norm(dim=-1) <= a1.norm(dim=-1) + 1e-10).all()


def test_rotated_is_conjugate_of_shift():
    en = _energy()
    b = build_base("linear_energy_shift", en, rho=4.0, dtype=F64)
    br = build_base("linear_energy_shift_rot", en, rho=4.0, dtype=F64, rot_seed=7)
    T = torch.tensor(random_rotation(b.n, 7))  # maps B1+ coordinates to rotated ones
    x0 = torch.randn(32, b.n, dtype=F64)
    m, v = b.transition_mean_var(x0, 1.0)
    mr, vr = br.transition_mean_var(x0 @ T.T, 1.0)
    assert (mr - m @ T.T).abs().max() < 1e-10
    cov = b.U64 @ torch.diag(v) @ b.U64.T
    covr = br.U64 @ torch.diag(vr) @ br.U64.T
    assert (covr - T @ cov @ T.T).abs().max() < 1e-10
    assert (br.theta64.sort().values - b.theta64.sort().values).abs().max() < 1e-10


@pytest.mark.parametrize("R", [2.0, 4.0, 8.0])
def test_iso_ou_matches_terminal_variance(R):
    en = _energy()
    ref = build_base("linear_energy_shift", en, rho=R, dtype=F64)
    iso = build_base("iso_ou", en, dtype=F64, iso_match_rho=R)
    assert iso.info["iso_theta"] > 0
    assert abs(terminal_var(iso) - terminal_var(ref)) < 1e-9
    assert iso.drift(0.0, torch.ones(1, en.n, dtype=F64)).abs().max() > 0  # f = -theta x != 0
    assert (iso.b64 == 0).all()


B0, B1, SIG = 20.0, 0.1, 1.0


def test_vp_matches_official():
    torch.manual_seed(0)
    B, n = 256, 6
    vp = VPBase(n, B0, B1, SIG, dtype=F64)
    x0, x1 = torch.randn(B, n, dtype=F64), 2 * torch.randn(B, n, dtype=F64)
    t, z = torch.rand(B, 1, dtype=F64), torch.randn(B, n, dtype=F64)
    assert (vp.bridge_sample(t, x0, x1, z=z)
            - vp_sample_posterior(t, x0, x1, B0, B1, SIG, z)).abs().max() < 1e-6
    ones = torch.ones(B, 1, dtype=F64)
    assert (vp.cm_target(x0, x1) - vp_cond_score(x0, ones, x1, B0, B1, SIG)).abs().max() < 1e-10
    a1 = torch.randn(B, n, dtype=F64)
    assert (vp.lean_adjoint(a1, t) - a1 * torch.exp(vp_coeff2(t, B0, B1))).abs().max() < 1e-10
    for tl in (0.0, 0.4, 0.99):
        assert abs(float(vp.diff(tl)) - SIG * (B0 + (B1 - B0) * tl) ** 0.5) < 1e-6


def test_vp_exact_step_vs_fine_euler():
    """Constant control: 10 exact steps == 2000 Euler-Maruyama steps in distribution."""
    torch.manual_seed(1)
    N, n = 40_000, 2
    vp = VPBase(n, B0, B1, SIG, dtype=F64)
    v = torch.tensor([[0.5, -1.0]], dtype=F64)
    x0 = torch.randn(N, n, dtype=F64)
    xa = x0.clone()
    for j in range(10):
        xa = vp.step(xa, j / 10, 0.1, v=v.expand(N, n))
    xb = x0.clone()
    K = 2000
    for j in range(K):
        t, dt = j / K, 1 / K
        g2 = SIG**2 * vp.beta(t)
        xb = xb + (vp.drift(t, xb) + g2 * v) * dt + (g2 * dt) ** 0.5 * torch.randn_like(xb)
    assert (xa.mean(0) - xb.mean(0)).abs().max() < 0.02
    assert ((xa.var(0) / xb.var(0)) - 1).abs().max() < 0.03


def test_vp_lean_adjoint_vs_autodiff():
    vp = VPBase(3, B0, B1, SIG, dtype=F64)
    steps, k = 50, 20
    x = torch.randn(4, 3, dtype=F64, requires_grad=True)
    y = x
    torch.manual_seed(2)
    for j in range(k, steps):
        y = vp.step(y, j / steps, 1 / steps, z=torch.randn(4, 3, dtype=F64))
    a1 = torch.randn(4, 3, dtype=F64)
    (g,) = torch.autograd.grad((y * a1).sum(), x)
    want = vp.lean_adjoint(a1, torch.full((4, 1), k / steps, dtype=F64))
    assert (g - want).abs().max() < 1e-10


@pytest.mark.parametrize("kind", ["zero", "linear_energy", "linear_energy_shift",
                                  "linear_energy_shift_rot", "iso_ou", "ve_geometric", "vp"])
def test_build_base_all_kinds(kind):
    en = _energy()
    b = build_base(kind, en, rho=2.0, sigma=1.0, dtype=F64, sigma_min=0.001, iso_match_rho=2.0)
    assert b.info["kind"] == kind
    x = torch.randn(8, en.n, dtype=F64)
    y = b.step(x, 0.3, 0.01, v=torch.randn_like(x))
    assert torch.isfinite(y).all()
    t = torch.rand(8, 1, dtype=F64)
    assert torch.isfinite(b.bridge_sample(t, x, y)).all()
    assert torch.isfinite(b.cm_target(x, y)).all()
