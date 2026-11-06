"""f = 0 reference functions ported from facebookresearch/adjoint_samplers
(components/sde.py: BrownianMotionSDE, VESDE, VPSDE, sdeint). Only change: the noise `z`
can be passed in so results are comparable sample-by-sample.
"""
import numpy as np
import torch


def bm_sample_posterior(t, x0, x1, sigma, z):
    mean = (1 - t) * x0 + t * x1
    var = (1 - t) * t * sigma**2
    var[var < 0] = 0
    noise = var.sqrt() * z
    return mean + noise


def bm_cond_score(x0, t, xt, sigma):
    loc, var = x0, sigma**2 * t
    return (loc - xt) / var


def em_controlled_step(x, dt, sigma, u, z):
    # ControlledSDE.drift = ref drift (0) + g^2 u ;  sdeint Euler-Maruyama
    drift = (sigma**2) * u * dt
    diffusion = sigma * dt**0.5 * z
    return x + drift + diffusion


# ---- VESDE (components/sde.py) used by group B0-S ---------------------------
def ve_diffsquare_integral(t, sigma_min, sigma_max):
    return (sigma_max**2) * (1 - (sigma_max / sigma_min) ** (-2 * t))


def ve_diff(t, sigma_min, sigma_max):
    r = sigma_max / sigma_min
    return sigma_min * (r ** (1 - t)) * ((2 * np.log(r)) ** 0.5)


def ve_sample_posterior(t, x0, x1, sigma_min, sigma_max, z):
    total_var = sigma_max**2 - sigma_min**2
    t_reparam = ve_diffsquare_integral(t, sigma_min, sigma_max) / total_var
    mean = (1 - t_reparam) * x0 + t_reparam * x1
    coeff = total_var * t_reparam * (1 - t_reparam)
    coeff[coeff < 0] = 0
    return mean + torch.sqrt(coeff) * z


def ve_cond_score(x0, t, xt, sigma_min, sigma_max):
    return (x0 - xt) / ve_diffsquare_integral(t, sigma_min, sigma_max)


def ve_em_controlled_step(x, t, dt, sigma_min, sigma_max, u, z):
    # ControlledSDE drift = 0 + g(t)^2 u ; sdeint uses the left point t of each step
    g = ve_diff(t, sigma_min, sigma_max)
    return x + (g**2) * u * dt + g * dt**0.5 * z


# ---- VPSDE (components/sde.py) used by the VP control group -----------------
def _vp_beta(t, b0, b1):
    return b0 + (b1 - b0) * t  # torch.lerp(beta0, beta1, t)


def vp_coeff2(t, b0, b1):  # = -1/2 int_t^1 beta
    return -0.25 * (1 - t) * (_vp_beta(t, b0, b1) + b1)


def vp_pt_gauss_param(t, mu0, b0, b1, sigma):
    coeff = -0.25 * t * (_vp_beta(t, b0, b1) + b0)
    mu = mu0 * torch.exp(coeff)
    var = sigma**2 * (1 - torch.exp(2 * coeff))
    return mu, var


def vp_cond_score(x0, t, xt, b0, b1, sigma):
    loc, var = vp_pt_gauss_param(t, x0, b0, b1, sigma)
    return (loc - xt) / var


def vp_sample_posterior(t, x0, x1, b0, b1, sigma, z):
    # beta0 / beta1 are registered tensor buffers in the official VPSDE
    b0, b1 = (torch.tensor(b, dtype=t.dtype) for b in (b0, b1))
    bt = _vp_beta(t, b0, b1)
    coeff1 = -0.25 * t * (bt + b0)
    coeff2 = -0.25 * (1 - t) * (bt + b1)
    coeff3 = -0.25 * (b1 + b0)
    mu = torch.exp(coeff1) * (1 - torch.exp(2 * coeff2)) / (1 - torch.exp(2 * coeff3)) * x0 \
        + torch.exp(coeff2) * (1 - torch.exp(2 * coeff1)) / (1 - torch.exp(2 * coeff3)) * x1
    var = (1 - torch.exp(2 * coeff1)) * (1 - torch.exp(2 * coeff2)) / (1 - torch.exp(2 * coeff3) + 1e-8)
    var[var < 0] = 0
    return mu + sigma * torch.sqrt(var) * z
