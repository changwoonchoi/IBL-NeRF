import torch
from typing import *
import numpy as np


def histogram(img: torch.Tensor, channels: List[int]=[32, 32, 32]) -> torch.Tensor:
    """
    Returns a color histogram of an input image

    Args:
        img: (H, W, 3)
        mask: (H, W)
        channels: List of length 3 containing number of bins per each channel
    Returns:
        hist: Histogram of shape (*channels)
    """
    tgt_img = img.clone().detach()
    max_rgb = torch.LongTensor([255] * 3).to(tgt_img.device)
    bin_size = torch.ceil(max_rgb.float() / torch.tensor(channels).float().to(tgt_img.device)).long()

    # when RGB value is in [0, 1]
    if tgt_img.max() <= 1:
        tgt_img = (tgt_img * max_rgb.reshape(-1, 3)).long()

    if len(img.shape) == 3:
        tgt_rgb = tgt_img.reshape(-1, 3).long()
        tgt_rgb = tgt_rgb // bin_size.reshape(-1, 3)
        tgt_rgb = tgt_rgb[:, 0] + channels[0] * tgt_rgb[:, 1] + channels[0] * channels[1] * tgt_rgb[:, 2]

        hist = torch.bincount(tgt_rgb, minlength=channels[0] * channels[1] * channels[2]).float()
        hist = hist.reshape(*channels)
    else:  # Batched input
        tgt_img = tgt_img // bin_size.reshape(-1, 3)
        tgt_img = tgt_img[..., 0] + channels[0] * tgt_img[..., 1] + channels[0] * channels[1] * tgt_img[..., 2]  # (B, H, W)
        tgt_img = tgt_img.reshape(tgt_img.shape[0], -1).long()  # (B, H * W)
        hist = torch.zeros([tgt_img.shape[0], channels[0] * channels[1] * channels[2]], device=tgt_img.device, dtype=torch.long).scatter_add(
            dim=-1, index=tgt_img, src=torch.ones_like(tgt_img, dtype=torch.long))
        hist = hist.reshape([hist.shape[0], *channels])

    return hist

