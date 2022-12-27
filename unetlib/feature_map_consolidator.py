import torch
import torch.nn.functional as F
from torch import nn

from unetlib.utils import default
from unetlib.resnet import Block


class FeatureMapConsolidator(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_ins=tuple(),
        dim_outs=tuple(),
        resize_fmap_before=True,
        conv_block_fn=None,
    ):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.needs_consolidating = len(dim_ins) > 0

        block_fn = default(conv_block_fn, Block)

        self.feature_map_convolutions = nn.ModuleList(
            [block_fn(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)]
        )
        self.resize_fmap_before = resize_fmap_before

        self.final_dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    @staticmethod
    def resize_fmaps(fmaps, target_size):
        return [
            F.interpolate(fmap, (fmap.shape[-3], target_size, target_size))
            for fmap in fmaps
        ]

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.needs_consolidating:
            return x

        if self.resize_fmap_before:
            fmaps = self.resize_fmaps(fmaps, target_size)

        outs = []
        for fmap, conv in zip(fmaps, self.feature_map_convolutions):
            outs.append(conv(fmap))

        if self.resize_fmap_before:
            outs = self.resize_fmaps(outs, target_size)

        return torch.cat((x, *outs), dim=1)
