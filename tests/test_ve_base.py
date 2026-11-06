"""Group B0-S: GeometricVEBase against the official VESDE (ported in asbs_reference)."""
import pytest
import torch

from asbs_reference import (ve_cond_score, ve_diff, ve_em_controlled_step,
                            ve_sample_posterior)
from sbmis.base.linear_energy_base import GeometricVEBase

SCHEDULES = [(0.001, 1.0), (0.001, 2.0), (0.01, 1.0)]


@pytest.mark.parametrize("smin,smax", SCHEDULES)
def test_ve_matches_official(smin, smax):
    torch.manual_seed(0)
    B, n = 256, 7
    base = GeometricVEBase(n, sigma_min=smin, sigma_max=smax, dtype=torch.float64)
    x0 = torch.randn(B, n, dtype=torch.float64)
    x1 = torch.randn(B, n, dtype=torch.float64) * 2
    t = torch.rand(B, 1, dtype=torch.float64)
    z = torch.randn(B, n, dtype=torch.float64)
    # bridge p(X_t | X_0, X_1)
    ours = base.bridge_sample(t, x0, x1, z=z)
    ref = ve_sample_posterior(t, x0, x1, smin, smax, z)
    assert (ours - ref).abs().max() < 1e-10
    # corrector target = cond_score(x0, t=1, x1)
    ones = torch.ones(B, 1, dtype=torch.float64)
    assert (base.cm_target(x0, x1) - ve_cond_score(x0, ones, x1, smin, smax)).abs().max() < 1e-10
    # diffusion coefficient; euler=True is the official left-point Euler step, the default
    # step injects exactly int_t^{t+dt} g^2
    v = torch.randn(B, n, dtype=torch.float64)
    euler = GeometricVEBase(n, sigma_min=smin, sigma_max=smax, dtype=torch.float64, euler=True)
    for tl in (0.0, 0.3, 0.99):
        assert abs(float(base.diff(tl)) - ve_diff(tl, smin, smax)) < 1e-10
        step = euler.step(x0, tl, 0.01, v=v, z=z)
        assert (step - ve_em_controlled_step(x0, tl, 0.01, smin, smax, v, z)).abs().max() < 1e-10
        d = smax**2 * ((smax / smin) ** (-2 * tl) - (smax / smin) ** (-2 * (tl + 0.01)))
        assert (base.step(x0, tl, 0.01, v=v, z=z) - (x0 + d * v + d**0.5 * z)).abs().max() < 1e-10
    # f = 0: lean adjoint is constant in t
    assert torch.equal(base.lean_adjoint(v, t), v)


@pytest.mark.parametrize("smin,smax", SCHEDULES)
@pytest.mark.parametrize("euler,steps", [(True, 2000), (False, 20), (False, 2000)])
def test_ve_total_variance(smin, smax, euler, steps):
    """Uncontrolled simulation injects sigma_max^2 - sigma_min^2 of variance: on any grid with
    the exact step, only on a fine grid with the official Euler step."""
    torch.manual_seed(1)
    base = GeometricVEBase(4, sigma_min=smin, sigma_max=smax, dtype=torch.float64, euler=euler)
    x = torch.zeros(40000, 4, dtype=torch.float64)
    for j in range(steps):
        x = base.step(x, j / steps, 1 / steps)
    want = smax**2 - smin**2
    assert abs(x.var(0).mean().item() / want - 1) < 0.03
