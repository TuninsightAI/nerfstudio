from __future__ import annotations

from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn


def _rgb_to_yiq(rgb_img):
    """
    Convert RGB image to YIQ.
    Assumes rgb_img is a PyTorch tensor of shape (batch_size, 3, height, width) and in [0, 1].
    """
    # Define the conversion matrix
    transformation_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ]).to(rgb_img.device)

    # Reshape the image tensor and perform matrix multiplication
    *b, channel = rgb_img.shape
    yiq_img = torch.matmul(transformation_matrix, rgb_img.T).T

    # Reshape the output back to the original shape
    return yiq_img.view(*b, channel)


def yiq_color_space_loss(rgb_img1: Float[Tensor, "*batch 3"], rg2_img2: Float[Tensor, "*batch 3"], *,
                         channel_weight: Tuple[float | int, float | int, float | int],
                         ):
    *b, c = rgb_img1.shape
    assert rg2_img2.shape == rgb_img1.shape
    hsv1, hsv2 = _rgb_to_yiq(rgb_img1), _rgb_to_yiq(rg2_img2)
    weight = torch.zeros(1, 3, device=torch.device("cuda"), dtype=torch.float)
    weight[0, :] = torch.Tensor(channel_weight).cuda().type(torch.float)
    return (torch.abs(hsv1 - hsv2) * weight).mean()


class YIQLoss(nn.Module):

    def __init__(self, channel_weight: Tuple[float, float, float], ) -> None:
        super().__init__()
        self.channel_weight = channel_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return yiq_color_space_loss(pred, target, channel_weight=self.channel_weight)
