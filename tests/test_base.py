"""Unit tests T1-T5, T7 (plan §6). T6 is in test_t6_toy.py (needs training)."""
import numpy as np
import pytest
import scipy.linalg
import torch

from asbs_reference import bm_cond_score, bm_sample_posterior, em_controlled_step
from sbmis.base.linear_energy_base import LinearEnergyBase
from sbmis.energies.mis import MISEnergy
from sbmis.graphs import er_graph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F64 = torch.float64


def small_energy(n=8, seed=3):
    return MISEnergy(er_graph(n, 0.4, seed), lambda_p=2.0, kappa_w=2.0, temp=0.5,
                     device=DEV, dtype=F64)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# ---------------------------------------------------------------- T1
def test_t1_grad_energy_matches_autograd():
    en = small_energy(12)
    x = torch.randn(64, en.n, dtype=F64, device=DEV, requires_grad=True)
    g_auto, = torch.autograd.grad(en.e_tgt(x).sum(), x)
    assert (g_auto - en.grad_tgt(x)).abs().max().item() < 1e-6
    gq, = torch.autograd.grad(en.e_quad(x).sum(), x)
    assert (gq - en.grad_quad(x)).abs().max().item() < 1e-6


# ---------------------------------------------------------------- T2
def test_t2_rho0_matches_official_f0():
    en = small_energy()
    sigma = 1.3
    base = LinearEnergyBase(en.M64, en.c64, sigma=sigma, rho=0.0, device=DEV, dtype=F64)
    B, n = 256, en.n
    g = torch.Generator(device=DEV).manual_seed(0)
    x0 = torch.randn(B, n, dtype=F64, device=DEV, generator=g)
    x1 = torch.randn(B, n, dtype=F64, device=DEV, generator=g)
    z = torch.randn(B, n, dtype=F64, device=DEV, generator=g)
    t = torch.rand(B, 1, dtype=F64, device=DEV, generator=g)
    # bridge
    ref = bm_sample_posterior(t, x0, x1, sigma, z)
    assert (base.bridge_sample(t, x0, x1, z=z) - ref).abs().max().item() < 1e-6
    # CM target
    ref = bm_cond_score(x0, torch.ones(B, 1, dtype=F64, device=DEV), x1, sigma)
    assert (base.cm_target(x0, x1) - ref).abs().max().item() < 1e-6
    # lean adjoint is constant
    assert (base.lean_adjoint(x1, t) - x1).abs().max().item() < 1e-12
    # integrator == Euler-Maruyama of the controlled f = 0 SDE
    u = torch.randn(B, n, dtype=F64, device=DEV, generator=g)
    ref = em_controlled_step(x0, 0.01, sigma, u, z)
    assert (base.step(x0, 0.3, 0.01, v=u, z=z) - ref).abs().max().item() < 1e-6
    # linear_energy with rho -> tiny also agrees (Taylor branch)
    base_t = LinearEnergyBase(en.M64, en.c64, sigma=sigma, rho=1e-9, device=DEV, dtype=F64)
    m_ref = (1 - t) * x0 + t * x1
    m, v = base_t.bridge_mean_var(t, x0, x1)
    assert (m - m_ref).abs().max().item() < 1e-6
    assert (v - sigma**2 * t * (1 - t)).abs().max().item() < 1e-6


# ---------------------------------------------------------------- T3
@pytest.mark.parametrize("rho", [-2.0, 1.0, 4.0])
def test_t3_transition_vs_euler_maruyama(rho):
    en = small_energy()
    sigma, n, N = 1.0, en.n, 40_000
    base = LinearEnergyBase(en.M64, en.c64, sigma=sigma, rho=rho, device=DEV, dtype=F64)
    xs = torch.linspace(-1, 1, n, dtype=F64, device=DEV)[None]
    torch.manual_seed(0)
    x = xs.expand(N, n).clone()
    dt, steps = 1e-4, 10_000
    for _ in range(steps):
        x = x + base.drift(None, x) * dt + sigma * dt**0.5 * torch.randn_like(x)
    mean, var_e = base.transition_mean_var(xs, 1.0)
    cov = base.U.to(F64) @ torch.diag(var_e) @ base.U.to(F64).T
    emp_mean = x.mean(0)
    emp_cov = torch.cov(x.T)
    assert (emp_mean - mean[0]).abs().max().item() < 0.02
    rel_cov = (torch.linalg.norm(emp_cov - cov) / torch.linalg.norm(cov)).item()
    assert rel_cov < 0.05, rel_cov


# ---------------------------------------------------------------- T4
def _independent_bridge(F, g, sigma, x0, x1, t):
    """Gaussian conditioning of the linear SDE dx = (F x + g) dt + sigma dW via expm."""
    n = F.shape[0]

    def moments(s):
        # mean via augmented expm, covariance via Van Loan
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n], aug[:n, n] = F, g
        E = scipy.linalg.expm(aug * s)
        Phi, r = E[:n, :n], E[:n, n]
        VL = np.zeros((2 * n, 2 * n))
        VL[:n, :n], VL[:n, n:], VL[n:, n:] = -F, sigma**2 * np.eye(n), F.T
        E2 = scipy.linalg.expm(VL * s)
        Sig = E2[n:, n:].T @ E2[:n, n:]
        return Phi, r, Sig

    Phi_t, r_t, S_t = moments(t)
    Phi_1, r_1, S_1 = moments(1.0)
    Phi_rem, _, _ = moments(1.0 - t)
    m_t, m_1 = Phi_t @ x0 + r_t, Phi_1 @ x0 + r_1
    C = Phi_rem @ S_t                      # Cov(x_1, x_t)
    K = C.T @ np.linalg.inv(S_1)
    return m_t + K @ (x1 - m_1), S_t - K @ C


@pytest.mark.parametrize("rho", [-2.0, 0.5, 4.0, 8.0])
def test_t4_bridge_vs_expm_conditioning(rho):
    en = small_energy()
    sigma = 0.8
    base = LinearEnergyBase(en.M64, en.c64, sigma=sigma, rho=rho, device="cpu", dtype=F64)
    F = -base.lam * en.M64.numpy()
    g = -base.lam * en.c64.numpy()
    rng = np.random.default_rng(0)
    for t in [0.05, 0.3, 0.5, 0.9]:
        x0, x1 = rng.normal(size=en.n), rng.normal(size=en.n)
        m_ref, C_ref = _independent_bridge(F, g, sigma, x0, x1, t)
        m, v = base.bridge_mean_var(torch.full((1, 1), t, dtype=F64),
                                    torch.tensor(x0)[None], torch.tensor(x1)[None])
        C = (base.U64 @ torch.diag(v[0]) @ base.U64.T).numpy()
        assert np.abs(m[0].numpy() - m_ref).max() / np.abs(m_ref).max() < 1e-6
        assert np.abs(C - C_ref).max() / np.abs(C_ref).max() < 1e-6
    # CM target = grad_{x1} log N(x1; transition mean, cov) with expm moments
    x0, x1 = rng.normal(size=en.n), rng.normal(size=en.n)
    aug = np.zeros((en.n + 1, en.n + 1))
    aug[:en.n, :en.n], aug[:en.n, en.n] = F, g
    E = scipy.linalg.expm(aug)
    mean1 = E[:en.n, :en.n] @ x0 + E[:en.n, en.n]
    VL = np.zeros((2 * en.n, 2 * en.n))
    VL[:en.n, :en.n], VL[:en.n, en.n:], VL[en.n:, en.n:] = -F, sigma**2 * np.eye(en.n), F.T
    E2 = scipy.linalg.expm(VL)
    S1 = E2[en.n:, en.n:].T @ E2[:en.n, en.n:]
    score_ref = -np.linalg.solve(S1, x1 - mean1)
    score = base.cm_target(torch.tensor(x0)[None], torch.tensor(x1)[None])[0].numpy()
    assert np.abs(score - score_ref).max() / np.abs(score_ref).max() < 1e-6


# ---------------------------------------------------------------- T5
@pytest.mark.parametrize("rho", [-2.0, 1.0, 4.0, 8.0])
def test_t5_lean_adjoint_vs_autodiff(rho):
    en = small_energy()
    base = LinearEnergyBase(en.M64, en.c64, sigma=1.0, rho=rho, device="cpu", dtype=F64)
    torch.manual_seed(0)
    B, steps = 4, 100
    ts = torch.linspace(0, 1, steps + 1, dtype=F64)
    zs = torch.randn(steps, B, en.n, dtype=F64)
    a1 = torch.randn(B, en.n, dtype=F64)
    for k in [0, 30, 77]:
        xk = torch.randn(B, en.n, dtype=F64, requires_grad=True)
        x = xk
        for j in range(k, steps):
            x = base.step(x, float(ts[j]), float(ts[j + 1] - ts[j]), v=None, z=zs[j])
        vjp, = torch.autograd.grad((x * a1).sum(), xk)
        a_t = base.lean_adjoint(a1, torch.full((B, 1), float(ts[k]), dtype=F64))
        expm = torch.tensor(scipy.linalg.expm(-base.lam * en.M64.numpy() * (1 - float(ts[k]))))
        assert rel(a_t, a1 @ expm.T) < 1e-6
        assert rel(vjp, a_t) < 1e-6


# ---------------------------------------------------------------- T7
@pytest.mark.parametrize("rho", [0.0, 4.0])
def test_t7_integrator_step_size_invariance(rho):
    en = small_energy()
    base = LinearEnergyBase(en.M64, en.c64, sigma=1.0, rho=rho, device=DEV, dtype=F64)
    N, n = 200_000, en.n
    u = torch.linspace(-1.0, 1.0, n, dtype=F64, device=DEV)[None]

    def run(steps, seed):
        torch.manual_seed(seed)
        x = torch.randn(N, n, dtype=F64, device=DEV)
        dt = 1.0 / steps
        for j in range(steps):
            x = base.step(x, j * dt, dt, v=u.expand(N, n))
        return x

    xa, xb = run(5, 1), run(500, 2)
    sd = xb.std(0)
    assert ((xa.mean(0) - xb.mean(0)).abs() / sd).max().item() < 0.02
    ca, cb = torch.cov(xa.T), torch.cov(xb.T)
    assert (torch.linalg.norm(ca - cb) / torch.linalg.norm(cb)).item() < 0.02


# ---------------------------------------------------------------- regression
def test_bridge_float32_t_plus_tau_consistency():
    """float32 t must not break t + tau = 1 inside the float64 G computation."""
    th = 1.0001e-4  # just above the Taylor threshold, where G's cancellation is worst
    base32 = LinearEnergyBase(np.array([[th]]), np.array([0.7]), sigma=1.0, lam=1.0,
                              dtype=torch.float32)
    t = torch.tensor([[1e-6], [0.3], [1 - 2**-20]], dtype=torch.float32)
    x0 = torch.full((3, 1), -0.4)
    x1 = torch.full((3, 1), 1.1)
    m32, _ = base32.bridge_mean_var(t, x0, x1)
    base64 = LinearEnergyBase(np.array([[th]]), np.array([0.7]), sigma=1.0, lam=1.0, dtype=F64)
    m64, _ = base64.bridge_mean_var(t.double(), x0.double(), x1.double())
    assert (m32.double() - m64).abs().max().item() < 1e-6
