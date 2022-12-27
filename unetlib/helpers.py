import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn

from unetlib.utils import default


def up_sample(dim, dim_out):
    return nn.ConvTranspose3d(dim, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def down_sample(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c f (h s1) (w s2) -> b (c s1 s2) f h w", s1=2, s2=2),
        nn.Conv3d(dim * 4, dim_out, 1),
    )


def kernel_and_same_pad(*kernel_size):
    paddings = tuple(map(lambda k: k // 2, kernel_size))
    return dict(kernel_size=kernel_size, padding=paddings)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.scale_squared = scale_factor**2
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * self.scale_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange(
                "b (c r s) f h w -> b c f (h r) (w s)", r=scale_factor, s=scale_factor
            ),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, *rest_dims = conv.weight.shape
        conv_weight = torch.empty(o // self.scale_squared, i, *rest_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o r) ...", r=self.scale_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = self.net(x)
        return x
