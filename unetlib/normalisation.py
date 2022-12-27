from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + eps).sqrt() * self.gamma


class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight

        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        p_var = partial(torch.var, unbiased=False)
        var = reduce(weight, "o ... -> o 1 1 1 1", p_var)
        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
