"""Noise schedule computation for diffusion models.

Ported from microscope/src/pipeline.cpp:27-39.
Implements the "scaled_linear" beta schedule from diffusers.
"""

import math
from dataclasses import dataclass


@dataclass
class NoiseParams:
    timestep: int
    alpha_cumprod: float
    sqrt_alpha: float
    sqrt_one_minus_alpha: float
    inv_sqrt_alpha: float


def compute_alpha_cumprod(timestep: int) -> float:
    """Compute cumulative product of alphas up to the given timestep.

    Diffusers "scaled_linear" schedule: beta_start=0.00085, beta_end=0.012.
    betas = linspace(sqrt(beta_start), sqrt(beta_end), 1000) ** 2
    """
    sqrt_beta_start = math.sqrt(0.00085)
    sqrt_beta_end = math.sqrt(0.012)
    alpha_cumprod = 1.0
    for t in range(timestep + 1):
        sqrt_beta = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * t / 999.0
        beta = sqrt_beta * sqrt_beta
        alpha_cumprod *= 1.0 - beta
    return alpha_cumprod


def compute_noise_params(model_type: str, strength: float = 0.5) -> NoiseParams:
    """Compute noise schedule parameters for the given model type.

    Args:
        model_type: "sdxs" or "sd-turbo"
        strength: Denoising strength (only used for sd-turbo), 0.0 to 1.0

    Returns:
        NoiseParams with precomputed schedule values
    """
    if model_type == "sdxs":
        # Euler scheduler, 1 step -> timestep 999
        timestep = 999
    else:
        # Default scheduler, 50 steps, strength-based
        t_idx = max(0, int(50 * (1.0 - strength)))
        # Timesteps are evenly spaced from 999 down
        timestep = 999 - t_idx * (1000 // 50)
        timestep = max(0, timestep)

    ap = compute_alpha_cumprod(timestep)
    sqrt_alpha = math.sqrt(ap)
    sqrt_one_minus_alpha = math.sqrt(1.0 - ap)
    inv_sqrt_alpha = 1.0 / sqrt_alpha

    return NoiseParams(
        timestep=timestep,
        alpha_cumprod=ap,
        sqrt_alpha=sqrt_alpha,
        sqrt_one_minus_alpha=sqrt_one_minus_alpha,
        inv_sqrt_alpha=inv_sqrt_alpha,
    )
