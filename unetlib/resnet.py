from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from unetlib.helpers import kernel_and_same_pad, PixelShuffleUpsample
from unetlib.normalisation import WeightStandardizedConv3d
from unetlib.utils import default, is_divisible


class Block(nn.Module):
    def __init__(
        self, dim, dim_out, groups=8, weight_standardize=False, frame_kernel_size=1
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)
        conv = nn.Conv3d if not weight_standardize else WeightStandardizedConv3d

        self.proj = conv(dim, dim_out, **kernel_conv_kwargs(3, 3))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        frame_kernel_size=1,
        nested_unet_depth=0,
        nested_unet_dim=32,
        weight_standardize=False,
    ):
        super().__init__()
        self.block1 = Block(
            dim,
            dim_out,
            groups=groups,
            weight_standardize=weight_standardize,
            frame_kernel_size=frame_kernel_size,
        )

        if nested_unet_depth > 0:
            self.block2 = NestedResidualUnet(
                dim_out,
                depth=nested_unet_depth,
                M=nested_unet_dim,
                frame_kernel_size=frame_kernel_size,
                weight_standardize=weight_standardize,
                add_residual=True,
            )
        else:
            self.block2 = Block(
                dim_out,
                dim_out,
                groups=groups,
                weight_standardize=weight_standardize,
                frame_kernel_size=frame_kernel_size,
            )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class NestedResidualUnet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        M=32,
        frame_kernel_size=1,
        add_residual=False,
        groups=4,
        skip_scale=2**-0.5,
        weight_standardize=False,
    ):
        super().__init__()

        self.depth = depth
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        conv = WeightStandardizedConv3d if weight_standardize else nn.Conv3d

        for ind in range(depth):
            is_first = ind == 0
            dim_in = dim if is_first else M

            down = nn.Sequential(
                conv(dim_in, M, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.GroupNorm(groups, M),
                nn.SiLU(),
            )

            up = nn.Sequential(
                PixelShuffleUpsample(2 * M, dim_in),
                nn.GroupNorm(groups, dim_in),
                nn.SiLU(),
            )

            self.downs.append(down)
            self.ups.append(up)

        self.mid = nn.Sequential(
            conv(M, M, **kernel_and_same_pad(frame_kernel_size, 3, 3)),
            nn.GroupNorm(groups, M),
            nn.SiLU(),
        )

        self.skip_scale = skip_scale
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        is_video = x.ndim == 5

        if self.add_residual:
            residual = default(residual, x.clone())

        *_, h, w = x.shape

        layers = len(self.ups)

        assert h == w, "only works with square images"
        assert is_divisible(
            h, 2 ** len(self.ups)
        ), f"dimension {h} must be divisible by {2 ** layers} ({layers} layers in nested unet)"
        assert (
            h % (2**self.depth)
        ) == 0, "the unet has too much depth for the image being passed in"

        # hiddens

        hiddens = []

        # unet

        for down in self.downs:
            x = down(x)
            hiddens.append(x.clone().contiguous())

        x = self.mid(x)

        for up in reversed(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_scale), dim=1)
            x = up(x)

        # adding residual

        if self.add_residual:
            x = x + residual
            x = F.silu(x)

        return x
