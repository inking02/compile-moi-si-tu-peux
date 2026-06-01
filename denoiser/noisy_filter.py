import torch


def add_haze(img, t_range=(0.3, 0.9), A_range=(0.6, 1.0)):
    """
    img: Tensor [3, 16, 16] in [0,1]
    """

    t = torch.empty(1).uniform_(*t_range)
    A = torch.empty(3, 1, 1).uniform_(*A_range)

    hazy = img * t + A * (1 - t)

    return hazy.clamp(0, 1)
