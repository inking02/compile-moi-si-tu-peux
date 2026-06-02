"""
File: noisy_filter.py

Description: This module contains image corruption helpers for denoising
experiments, including a simple haze model based on atmospheric scattering.
"""

import torch


def add_haze(img, t_range=(0.3, 0.9), A_range=(0.6, 1.0)):
    """
    Simulates haze on an image using the atmospheric scattering model.

    Args:
        img (torch.Tensor): RGB image tensor in ``[0, 1]`` shaped ``(3, H, W)``.
        t_range (tuple[float, float]): Range for random transmission values.
        A_range (tuple[float, float]): Range for random atmospheric light values.

    Returns:
        torch.Tensor: Hazy image tensor clamped to ``[0, 1]``.
    """

    t = torch.empty(1).uniform_(*t_range)
    A = torch.empty(3, 1, 1).uniform_(*A_range)

    hazy = img * t + A * (1 - t)

    return hazy.clamp(0, 1)
