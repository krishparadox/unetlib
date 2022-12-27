from einops import rearrange
from torch import nn, einsum

from unetlib.normalisation import LayerNorm


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1, bias=False)

    def forward(self, x):
        f, h, w = x.shape[-3:]

        residual = x.clone()

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) ... -> b h (...) c", h=self.heads),
            (q, k, v),
        )

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h (f x y) d -> b (h d) f x y", f=f, x=h, y=w)
        return self.to_out(out) + residual
