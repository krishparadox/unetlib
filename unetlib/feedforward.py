from torch import nn
from unetlib.normalisation import Residual, LayerNorm


def FeedForward(dim, mult=4.0):
    inner_dim = int(dim * mult)
    return Residual(
        nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, inner_dim, 1, bias=False),
            nn.GELU(),
            LayerNorm(inner_dim),
            nn.Conv3d(inner_dim, dim, 1, bias=False),
        )
    )
