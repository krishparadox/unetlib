from functools import partial

import torch
from einops import rearrange
from torch import nn

from unetlib.utils import default, cast_to_tuple
from unetlib.helpers import kernel_and_same_pad, down_sample, up_sample
from unetlib.convnext import ConvNextBlock
from unetlib.resnet import ResnetBlock
from unetlib.transformer import TransformerBlock
from unetlib.attention import Attention
from unetlib.feature_map_consolidator import FeatureMapConsolidator


class BuildUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        frame_kernel_size=1,
        dim_mults=(1, 2, 4, 8),
        num_blocks_per_stage=(2, 2, 2, 2),
        num_self_attn_per_stage=(0, 0, 0, 1),
        nested_unet_depths=(0, 0, 0, 0),
        nested_unet_dim=32,
        channels=3,
        use_convnext=False,
        resnet_groups=8,
        consolidate_upsample_fmaps=True,
        skip_scale=2**-0.5,
        weight_standardize=False,
        attn_heads=8,
        attn_dim_head=32,
    ):
        super().__init__()

        self.train_as_images = frame_kernel_size == 1

        self.skip_scale = skip_scale
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(
            channels, init_dim, **kernel_and_same_pad(frame_kernel_size, 7, 7)
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        blocks = (
            partial(ConvNextBlock, frame_kernel_size=frame_kernel_size)
            if use_convnext
            else partial(
                ResnetBlock,
                groups=resnet_groups,
                weight_standardize=weight_standardize,
                frame_kernel_size=frame_kernel_size,
            )
        )
        nested_unet_depths = cast_to_tuple(nested_unet_depths, num_resolutions)
        num_blocks_per_stage = cast_to_tuple(num_blocks_per_stage, num_resolutions)
        assert all([num_blocks > 0 for num_blocks in num_blocks_per_stage])
        num_self_attn_per_stage = cast_to_tuple(
            num_self_attn_per_stage, num_resolutions
        )
        assert all(
            [
                num_self_attn_blocks >= 0
                for num_self_attn_blocks in num_self_attn_per_stage
            ]
        )
        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head)
        skip_dims = []

        down_stage_parameters = [
            in_out,
            nested_unet_depths,
            num_blocks_per_stage,
            num_self_attn_per_stage,
        ]

        up_stage_parameters = [
            reversed(params[:-1]) for params in down_stage_parameters
        ]
        for ind, (
            (dim_in, dim_out),
            nested_unet_depth,
            num_blocks,
            self_attn_blocks,
        ) in enumerate(zip(*down_stage_parameters)):
            skip_dims.append(dim_in)

            self.downs.append(
                nn.ModuleList(
                    [
                        blocks(
                            dim_in,
                            dim_in,
                            nested_unet_depth=nested_unet_depth,
                            nested_unet_dim=nested_unet_dim,
                        ),
                        nn.ModuleList(
                            [
                                blocks(
                                    dim_in,
                                    dim_in,
                                    nested_unet_depth=nested_unet_depth,
                                    nested_unet_dim=nested_unet_dim,
                                )
                                for _ in range(num_blocks - 1)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    dim_in, depth=self_attn_blocks, **attn_kwargs
                                )
                                for _ in range(self_attn_blocks)
                            ]
                        ),
                        down_sample(dim_in, dim_out),
                    ]
                )
            )
        mid_dim = dims[-1]
        mid_nested_unet_depth = nested_unet_depths[-1]

        self.mid = blocks(
            mid_dim,
            mid_dim,
            nested_unet_depth=mid_nested_unet_depth,
            nested_unet_dim=nested_unet_dim,
        )
        self.mid_attn = Attention(mid_dim)
        self.mid_after = blocks(
            mid_dim,
            mid_dim,
            nested_unet_depth=mid_nested_unet_depth,
            nested_unet_dim=nested_unet_dim,
        )

        self.mid_upsample = up_sample(mid_dim, dims[-2])

        for ind, (
            (dim_in, dim_out),
            nested_unet_depth,
            num_blocks,
            self_attn_blocks,
        ) in enumerate(zip(*up_stage_parameters)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        blocks(
                            dim_out + skip_dims.pop(),
                            dim_out,
                            nested_unet_depth=nested_unet_depth,
                            nested_unet_dim=nested_unet_dim,
                        ),
                        nn.ModuleList(
                            [
                                blocks(
                                    dim_out,
                                    dim_out,
                                    nested_unet_depth=nested_unet_depth,
                                    nested_unet_dim=nested_unet_dim,
                                )
                                for _ in range(num_blocks - 1)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    dim_out, depth=self_attn_blocks, **attn_kwargs
                                )
                                for _ in range(self_attn_blocks)
                            ]
                        ),
                        up_sample(dim_out, dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)

        if consolidate_upsample_fmaps:
            self.consolidator = FeatureMapConsolidator(
                dim,
                dim_ins=tuple(map(lambda m: dim * m, dim_mults)),
                dim_outs=(dim,) * len(dim_mults),
                conv_block_fn=blocks,
            )
        else:
            self.consolidator = FeatureMapConsolidator(dim=dim)

        final_dim_in = self.consolidator.final_dim_out

        self.final_conv = nn.Sequential(
            blocks(final_dim_in + dim, dim),
            nn.Conv3d(dim, out_dim, **kernel_and_same_pad(frame_kernel_size, 3, 3)),
        )

    def forward(self, x):
        is_image = x.ndim == 4
        assert not (
            is_image and not self.train_as_images
        ), "you specified a frame kernel size for the convolutions in this unet, but you are passing in images"
        assert not (not is_image and self.train_as_images), (
            "you specified no frame kernel size dimension, yet you are passing in a video. fold the frame dimension "
            "into the batch"
        )

        if is_image:
            x = rearrange(x, "b c h w -> b c 1 h w")

        x = self.init_conv(x)
        r = x.clone()
        down_hiddens = []
        up_hiddens = []

        for init_block, blocks, attn_blocks, downsample in self.downs:
            x = init_block(x)
            for block in blocks:
                x = block(x)
            for attn_block in attn_blocks:
                x = attn_block(x)
            down_hiddens.append(x)
            x = downsample(x)

        x = self.mid(x)
        x = self.mid_attn(x) + x
        x = self.mid_after(x)

        up_hiddens.append(x)
        x = self.mid_upsample(x)

        for init_block, blocks, attn_blocks, upsample in self.ups:
            x = torch.cat((x, down_hiddens.pop() * self.skip_scale), dim=1)
            x = init_block(x)
            for block in blocks:
                x = block(x)
            for attn_block in attn_blocks:
                x = attn_block(x)

            up_hiddens.insert(0, x)
            x = upsample(x)

        x = self.consolidator(x, up_hiddens)
        x = torch.cat((x, r), dim=1)
        out = self.final_conv(x)

        if is_image:
            out = rearrange(out, "b c 1 h w -> b c h w")

        return out
