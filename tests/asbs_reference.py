"""f = 0 reference functions ported from facebookresearch/adjoint_samplers
(components/sde.py: BrownianMotionSDE, sdeint). Only change: the noise `z`
can be passed in so results are comparable sample-by-sample.
"""
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
