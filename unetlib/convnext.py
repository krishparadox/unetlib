from functools import partial

from torch import nn

from unetlib.normalisation import LayerNorm
from unetlib.helpers import kernel_and_same_pad
from unetlib.resnet import NestedResidualUnet


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        mult=2,
        frame_kernel_size=1,
        nested_unet_depth=0,
        nested_unet_dim=32,
    ):
        super().__init__()
        kernel_conv_kwargs = partial(kernel_and_same_pad, frame_kernel_size)

        self.ds_conv = nn.Conv3d(dim, dim, **kernel_conv_kwargs(7, 7), groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv3d(dim, dim_out * mult, **kernel_conv_kwargs(3, 3)),
            nn.GELU(),
            nn.Conv3d(dim_out * mult, dim_out, **kernel_conv_kwargs(3, 3)),
        )

        self.nested_unet = (
            NestedResidualUnet(
                dim_out, depth=nested_unet_depth, M=nested_unet_dim, add_residual=True
            )
            if nested_unet_depth > 0
            else nn.Identity()
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        h = self.net(h)
        h = self.nested_unet(h)
        return h + self.res_conv(x)
