"""f = 0 reference functions ported from facebookresearch/adjoint_samplers
(components/sde.py: BrownianMotionSDE, VESDE, sdeint). Only change: the noise `z`
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
