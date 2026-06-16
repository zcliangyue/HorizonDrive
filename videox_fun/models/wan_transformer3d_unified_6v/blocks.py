# Transformer blocks for Wan Transformer

import torch
import torch.nn as nn
import math
from einops import rearrange

from .norms import WanLayerNorm
from .attention import WanSelfAttention, WAN_CROSSATTENTION_CLASSES

try:
    import fastvideo.envs as envs
except ModuleNotFoundError:
    class _FastVideoEnvs:
        FASTVIDEO_ATTENTION_BACKEND = None

    envs = _FastVideoEnvs()

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 prefix=""):
        super().__init__()
        self.prefix = prefix
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, prefix=prefix)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        pos_emb_fn,
        context=None,
        context_lens=None,
        dtype=torch.float32,
        num_views=1,
        cross_view_flex_attn=None,
        crossview_attn_type="full",
        cond_mask=None,
        kv_cache=None,
        return_kv=False,
        skip_cross_attn=False,
        score_vis_dir=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, T, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=-2)  # [B, 6/1, C] or [B, F, 6/1, C]

        if len(e[0].shape) != 3 and len(e[0].shape) != 4:
            raise ValueError(f"invalid e shape: {[ei.shape for ei in e]}")
        if len(x.shape) != 3:
            raise ValueError(f"invalid x shape: {x.shape}")

        time_invariant_modulation = (len(e[0].shape) == 4)

        def _unflatten(x):
            return rearrange(x, "b (f h w) c -> b f (h w) c", f=grid_sizes[0][0], h=grid_sizes[0][1], w=grid_sizes[0][2]) if time_invariant_modulation else x

        def _flatten(x):
            return rearrange(x, "b f hw c -> b (f hw) c") if time_invariant_modulation else x

        # self-attention
        temp_x = _unflatten(self.norm1(x)) * (1 + e[1]) + e[0]
        temp_x = _flatten(temp_x)
        temp_x = temp_x.to(dtype)

        attn_cond_mask = None if envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN" else cond_mask
        y = self.self_attn(
            temp_x,
            seq_lens,
            grid_sizes,
            pos_emb_fn,
            dtype,
            num_views=num_views,
            cross_view_flex_attn=cross_view_flex_attn,
            crossview_attn_type=crossview_attn_type,
            cond_mask=attn_cond_mask,
            kv_cache=kv_cache,
            return_kv=return_kv,
            score_vis_dir=score_vis_dir,
        )

        if return_kv:
            y, kv_cache = y
        x = x + _flatten(_unflatten(y) * e[2])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, skip_cross_attn):
            if not skip_cross_attn:
                x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, grid_sizes, num_views=num_views, cond_mask=attn_cond_mask)

            # ffn function
            temp_x = _unflatten(self.norm2(x)) * (1 + e[4]) + e[3]
            temp_x = _flatten(temp_x)
            temp_x = temp_x.to(dtype)
            y = self.ffn(temp_x)
            x = x + _flatten(_unflatten(y) * e[5])
            return x
        if context is not None:
            x = cross_attn_ffn(x, context, context_lens, e, skip_cross_attn)
        x = x.to(dtype)
        if return_kv:
            return x, kv_cache
        else:
            return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C] or [B, T, C]
        """
        e = (self.modulation + e.unsqueeze(-2)).chunk(2, dim=-2)  # [B, 2/1, C] or [B, T, 2/1, C]
        if len(e[0].shape) != 3 and len(e[0].shape) != 4:
            raise ValueError(f"invalid e shape: {[ei.shape for ei in e]}")
        if len(x.shape) != 3:
            raise ValueError(f"invalid x shape: {x.shape}")
        time_invariant_modulation = (len(e[0].shape) == 4)

        def _unflatten(x):
            return rearrange(x, "b (f h w) c -> b f (h w) c", f=grid_sizes[0][0], h=grid_sizes[0][1], w=grid_sizes[0][2]) if time_invariant_modulation else x

        def _flatten(x):
            return rearrange(x, "b f hw c -> b (f hw) c") if time_invariant_modulation else x

        dtype = x.dtype
        x = _flatten(_unflatten(self.norm(x)) * (1 + e[1]) + e[0])
        x = self.head(x.to(dtype))
        return x
