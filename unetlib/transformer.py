from torch import nn

from unetlib.attention import Attention
from unetlib.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, dim, *, depth, **kwargs):
        super().__init__()
        self.attn = Attention(dim, **kwargs)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x
