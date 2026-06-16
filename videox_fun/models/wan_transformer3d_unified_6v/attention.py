# Attention modules for Wan Transformer
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import functools
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import flex_attention

from .norms import WanRMSNorm
from .embeddings import sinusoidal_embedding_batchwise

try:
    import fastvideo.envs as envs
except ModuleNotFoundError:
    class _FastVideoEnvs:
        FASTVIDEO_ATTENTION_BACKEND = None

    envs = _FastVideoEnvs()

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major>=9:
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False

USE_PPU = os.environ.get("USE_PPU", "false").lower() in ("true", "1", "t")
FASTVIDEO_AVAILABLE = False

if not USE_PPU:
    try:
        from fastvideo.attention import DistributedAttention_VSA
        from fastvideo.platforms import AttentionBackendEnum
        from horizondrive.utils.block_attention import (
            cross_view_spatial_attn_block_mask,
            cross_view_spatial_attn_full_singleview_attn_block_mask,
            multi_view_full_attn_block_mask,
            single_view_full_attn_block_mask,
            cross_view_text_block_mask,
            block_sparse_attention,
            BLOCK_SIZE,
        )
        FASTVIDEO_AVAILABLE = True
    except ModuleNotFoundError:
        DistributedAttention_VSA = None
        block_sparse_attention = None
        cross_view_spatial_attn_block_mask = None
        cross_view_spatial_attn_full_singleview_attn_block_mask = None
        multi_view_full_attn_block_mask = None
        single_view_full_attn_block_mask = None
        cross_view_text_block_mask = None
        BLOCK_SIZE = 128

        class AttentionBackendEnum:
            SLIDING_TILE_ATTN = "SLIDING_TILE_ATTN"
            SAGE_ATTN = "SAGE_ATTN"
            FLASH_ATTN = "FLASH_ATTN"
            TORCH_SDPA = "TORCH_SDPA"
            VIDEO_SPARSE_ATTN = "VIDEO_SPARSE_ATTN"
            VMOBA_ATTN = "VMOBA_ATTN"
            SAGE_ATTN_THREE = "SAGE_ATTN_THREE"



# ---------- Multi-view map ----------
'''
0: camera_front
1: camera_front_left
2: camera_front_right
3: camera_rear_right
4: camera_rear
5: camera_rear_left
'''
mv_map = {
    0: [0, 1, 2],
    1: [0, 1, 5],
    2: [0, 2, 3],
    3: [2, 3, 4],
    4: [3, 4, 5],
    5: [1, 4, 5],
}


# ---------- Attention functions ----------

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
):
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = sageattn(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    elif attention_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    return out


# ---------- CrossView Mask / Flex Attention ----------

BLOCK_SIZE_FLEX = 128

class CrossViewMask:
    def __init__(self):
        pass

    def _generate_view_ids(self, grid_sizes, num_views, device):
        tokens_per_view = torch.prod(grid_sizes[0]).item() // num_views
        view_ids = torch.repeat_interleave(
            torch.arange(num_views),
            tokens_per_view
        )
        return view_ids.to(device)

    def get_view_mask(self, num_views, device):
        view_mask = torch.zeros((num_views, num_views), dtype=torch.bool, device=device)
        for target_id in range(num_views):
            source_ids = mv_map[target_id]
            for source_id in source_ids:
                view_mask[target_id, source_id] = True
        return view_mask

    def compute_sum_blocks(
        self,
        seq_len,
        device,
        mask=None,
    ):
        def _round_up_to_multiple(x, multiple):
            return (x + multiple - 1) // multiple * multiple
        Q_LEN = _round_up_to_multiple(seq_len, BLOCK_SIZE_FLEX)
        num_blocks = Q_LEN // BLOCK_SIZE_FLEX

        n = mask.shape[0]
        s = seq_len // n

        sum_blocks = torch.zeros((num_blocks, num_blocks), dtype=torch.int32, device=device)
        for q_frame_id in range(n):
            for kv_frame_id in range(n):
                if mask[q_frame_id, kv_frame_id]:
                    q_token_start = q_frame_id * s
                    q_token_end = (q_frame_id + 1) * s

                    kv_token_start = kv_frame_id * s
                    kv_token_end = (kv_frame_id + 1) * s

                    q_start_block = q_token_start // BLOCK_SIZE_FLEX
                    q_end_block = min(q_token_end // BLOCK_SIZE_FLEX + 1, num_blocks)

                    kv_start_block = kv_token_start // BLOCK_SIZE_FLEX
                    kv_end_block = min(kv_token_end // BLOCK_SIZE_FLEX + 1, num_blocks)

                    for q_block in range(q_start_block, q_end_block):
                        for kv_block in range(kv_start_block, kv_end_block):
                            q_block_start = max(q_block * BLOCK_SIZE_FLEX, q_token_start)
                            q_block_end = min((q_block + 1) * BLOCK_SIZE_FLEX, q_token_end)

                            kv_block_start = max(kv_block * BLOCK_SIZE_FLEX, kv_token_start)
                            kv_block_end = min((kv_block + 1) * BLOCK_SIZE_FLEX, kv_token_end)
                            if q_block_start < q_block_end and kv_block_start < kv_block_end:
                                overlap_q = q_block_end - q_block_start
                                overlap_kv = kv_block_end - kv_block_start
                                sum_blocks[q_block, kv_block] += overlap_q * overlap_kv

        full_blocks = sum_blocks == (BLOCK_SIZE_FLEX * BLOCK_SIZE_FLEX)
        partial_blocks = (sum_blocks > 0) & (sum_blocks < (BLOCK_SIZE_FLEX * BLOCK_SIZE_FLEX))
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)

        return partial_blocks, full_blocks

    def get_block_mask(self, num_views, seq_lens, grid_sizes, device):
        seq_lens = seq_lens.to(device)
        grid_sizes = grid_sizes.to(device)

        seq_len = seq_lens[0]
        view_mask = self.get_view_mask(num_views, device)
        partial_block_mask, full_block_mask = self.compute_sum_blocks(
            seq_len=seq_len.item(),
            device=device,
            mask=view_mask,
        )
        partial_block_mask = partial_block_mask.unsqueeze(0).unsqueeze(0)
        full_block_mask = full_block_mask.unsqueeze(0).unsqueeze(0)

        view_ids = self._generate_view_ids(grid_sizes, num_views, device)
        def sparse_mask_fn(b_idx, head_idx, q_idx, kv_idx):
            valid_indices = (q_idx < seq_len) & (kv_idx < seq_len)

            safe_q_idx = torch.where(valid_indices, q_idx, 0)
            safe_kv_idx = torch.where(valid_indices, kv_idx, 0)

            q_view = view_ids[safe_q_idx]
            kv_view = view_ids[safe_kv_idx]
            mask_val = view_mask[q_view, kv_view]

            return torch.where(valid_indices, mask_val, False)

        block_mask = flex_attention._create_sparse_block_from_block_mask(
            (partial_block_mask, full_block_mask),
            sparse_mask_fn,
            seq_lengths=(seq_len.item(), seq_len.item()),
            Q_BLOCK_SIZE=BLOCK_SIZE_FLEX,
            KV_BLOCK_SIZE=BLOCK_SIZE_FLEX,
        )
        return block_mask


kernel_options = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, block_mask=None):
    return flex_attention.flex_attention(q, k, v, block_mask=block_mask, kernel_options=kernel_options)


if not USE_PPU and FASTVIDEO_AVAILABLE:
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
            AttentionBackendEnum.SLIDING_TILE_ATTN, AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN, AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SAGE_ATTN_THREE)


# ---------- Block Sparse mask registry ----------

BLOCK_SPARSE_MASK_REGISTRY = {}
if not USE_PPU and FASTVIDEO_AVAILABLE:
    BLOCK_SPARSE_MASK_REGISTRY = {
        "block-sparse__view-wise-attn": multi_view_full_attn_block_mask,
        "block-sparse__spatial-crossview-attn": cross_view_spatial_attn_block_mask,
        "block-sparse__full-singleview-attn": single_view_full_attn_block_mask,
        "block-sparse__spatial-crossview-attn-full-singleview-attn": cross_view_spatial_attn_full_singleview_attn_block_mask,
    }


# ---------- Dispatched attention by crossview_attn_type ----------

def _multiview_attention(q, k, v, attn_type, num_views, grid_sizes,
                         seq_lens, window_size, cross_view_flex_attn, cond_mask):
    """Dispatch multi-view attention based on attn_type string."""
    b = q.shape[0]
    nf = grid_sizes[0, 0].item()
    f = nf // num_views

    # --- block sparse ---
    # block-sparse__view-wise-attn
    # block-sparse__spatial-crossview-attn
    # block-sparse__full-singleview-attn
    # block-sparse__spatial-crossview-attn-full-singleview-attn
    if attn_type in BLOCK_SPARSE_MASK_REGISTRY:
        if attn_type != "block-sparse__view-wise-attn":
            assert not USE_PPU, f"{attn_type} is not compatible with PPU."
        nf, h, w = grid_sizes[0].tolist()
        block_mask = BLOCK_SPARSE_MASK_REGISTRY[attn_type](
            num_views=num_views,
            num_frames_each_view=nf // num_views,
            num_tokens_each_frame=h * w,
            cond_mask=cond_mask[0] if cond_mask is not None else None,
            device_str=str(q.device),
        )
        return block_sparse_attention(q=q, k=k, v=v, block_mask=block_mask)

    # --- rearrange-based ---
    # full
    # spatial-crossview-attn
    # full-singleview-attn
    if attn_type == "full":
        return attention(q=q, k=k, v=v, k_lens=seq_lens, window_size=window_size)

    if attn_type == "spatial-crossview-attn":
        q = rearrange(q, 'b (nv f s) ... -> (b f) (nv s) ...', nv=num_views, f=f)
        k = rearrange(k, 'b (nv f s) ... -> (b f) (nv s) ...', nv=num_views, f=f)
        v = rearrange(v, 'b (nv f s) ... -> (b f) (nv s) ...', nv=num_views, f=f)
        x = attention(q, k, v)
        return rearrange(x, '(b f) (nv s) ... -> b (nv f s) ...', b=b, nv=num_views, f=f)

    if attn_type == "full-singleview-attn":
        q = rearrange(q, 'b (nv f s) ... -> (b nv) (f s) ...', nv=num_views, f=f)
        k = rearrange(k, 'b (nv f s) ... -> (b nv) (f s) ...', nv=num_views, f=f)
        v = rearrange(v, 'b (nv f s) ... -> (b nv) (f s) ...', nv=num_views, f=f)
        x = attention(q, k, v)
        return rearrange(x, '(b nv) (f s) ... -> b (nv f s) ...', b=b, nv=num_views, f=f)

    # --- mv_map loop ---
    # loop
    # view-wise-attn
    if attn_type in ("loop", "view-wise-attn"):
        q = rearrange(q, 'b (nv s) ... -> b nv s ...', nv=num_views)
        k = rearrange(k, 'b (nv s) ... -> b nv s ...', nv=num_views)
        v = rearrange(v, 'b (nv s) ... -> b nv s ...', nv=num_views)
        xs = []
        for tid in range(num_views):
            sids = mv_map[tid]
            cur_x = attention(
                q=q[:, tid],
                k=rearrange(k[:, sids], 'b n s ... -> b (n s) ...', n=len(sids)),
                v=rearrange(v[:, sids], 'b n s ... -> b (n s) ...', n=len(sids)),
                k_lens=(seq_lens // num_views) * len(sids),
                window_size=window_size,
            )
            xs.append(cur_x)
        return rearrange(torch.stack(xs, dim=1), 'b nv s ... -> b (nv s) ...', nv=num_views)

    # --- flex attention ---
    if attn_type == "flex":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return cross_view_flex_attn(q, k, v).transpose(1, 2)

    raise ValueError(f"Unknown multiview attn type: {attn_type}")


def _build_causal_mask(attn_type, grid_sizes, device):
    """Build temporal causal attn_mask for single-view modes. Returns None if no mask needed."""
    if attn_type == "blockwise_causal":
        n_frames, n_height, n_width = grid_sizes[0].tolist()
        n_tokens_per_img = n_height * n_width
        temp_mask = torch.ones(n_frames, n_frames, dtype=torch.bool, device=device).tril(diagonal=0)
        temp_mask = rearrange(temp_mask, 'i j -> i 1 j 1')
        temp_mask = temp_mask.repeat(1, n_tokens_per_img, 1, n_tokens_per_img)
        return rearrange(temp_mask, 'i j k l -> (i j) (k l)')

    if attn_type == "window_causal":
        n_frames, n_height, n_width = grid_sizes[0].tolist()
        n_tokens_per_img = n_height * n_width
        win = 20  # TODO: fix magic number, 80frames, 8s
        temp_mask = torch.ones(n_frames, n_frames, dtype=torch.bool, device=device).tril_(-win).logical_not_().tril_(0)
        temp_mask = rearrange(temp_mask, 'i j -> i 1 j 1')
        temp_mask = temp_mask.repeat(1, n_tokens_per_img, 1, n_tokens_per_img)
        return rearrange(temp_mask, 'i j k l -> (i j) (k l)')

    return None


def _singleview_attention_with_cond_mask(q, k, v, seq_lens, window_size,
                                         cond_mask, grid_sizes, attn_kwargs):
    """Single-view attention with cond/noise frame splitting."""
    b = q.shape[0]
    n, d = q.shape[2], q.shape[3]
    n_frames, n_height, n_width = grid_sizes[0].tolist()
    n_tokens_per_img = n_height * n_width

    cond_seq_idxs = cond_mask[0].nonzero(as_tuple=False).squeeze(-1)
    noncond_seq_idxs = (~cond_mask[0]).nonzero(as_tuple=False).squeeze(-1)
    q = rearrange(q, 'b (f s) ... -> b f s ...', f=n_frames, s=n_tokens_per_img)
    k = rearrange(k, 'b (f s) ... -> b f s ...', f=n_frames, s=n_tokens_per_img)
    v = rearrange(v, 'b (f s) ... -> b f s ...', f=n_frames, s=n_tokens_per_img)

    q_cond = q[:, cond_seq_idxs].contiguous().view(b, -1, n, d)
    k_cond = k[:, cond_seq_idxs].contiguous().view(b, -1, n, d)
    v_cond = v[:, cond_seq_idxs].contiguous().view(b, -1, n, d)
    x_cond = attention(
        q=q_cond, k=k_cond, v=v_cond,
        k_lens=seq_lens.new_full((b,), cond_seq_idxs.shape[0] * n_tokens_per_img),
        window_size=window_size, **attn_kwargs,
    )
    q_noise = q[:, noncond_seq_idxs].contiguous().view(b, -1, n, d)
    x_noise = attention(
        q=q_noise, k=k.view(b, -1, n, d), v=v.view(b, -1, n, d),
        k_lens=seq_lens, window_size=window_size, **attn_kwargs,
    )
    x = torch.zeros((b, n_frames, n_tokens_per_img, n, d), device=x_cond.device, dtype=x_cond.dtype)
    x[:, cond_seq_idxs] = x_cond.view(b, cond_seq_idxs.shape[0], n_tokens_per_img, n, d)
    x[:, noncond_seq_idxs] = x_noise.view(b, noncond_seq_idxs.shape[0], n_tokens_per_img, n, d)
    return x.view(b, -1, n, d).contiguous()


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 prefix=""):
        assert dim % num_heads == 0
        super().__init__()
        self.prefix = prefix
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


        if not USE_PPU and FASTVIDEO_AVAILABLE and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            self.to_gate_compress = nn.Linear(dim, dim)
            nn.init.zeros_(self.to_gate_compress.weight)
            nn.init.zeros_(self.to_gate_compress.bias)

            self.attn1 = DistributedAttention_VSA(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=_supported_attention_backends,
                prefix=f"{prefix}.attn1")

    def save_attention_map(self, q, k, grid_sizes, num_views, save_dir, layer_idx, crossview_attn_type="full"):
        import os
        import math

        # 1. Dimensions
        # q, k: [B, S, N, D]
        b, s, n, d = q.shape
        F_total, H, W = grid_sizes[0].tolist() # (F, H, W) in latent space
        seq_len = k.shape[1]

        # 2. Define Query Points (Start, Mid, End per View; 4x4 Spatial)
        F_per_view = F_total // num_views
        target_frames = [F_per_view // 2]

        # Spatial Grid (4x4)
        h_points = torch.linspace(H // 8, H - (H // 8), steps=4).long()
        w_points = torch.linspace(W // 8, W - (W // 8), steps=4).long()

        query_indices = []

        tokens_per_view = F_per_view * H * W
        tokens_per_frame = H * W

        # Calculate global indices for Q
        for v in range(num_views):
            for f_local in target_frames:
                for y in h_points:
                    for x in w_points:
                        # Global Index
                        idx = v * tokens_per_view + f_local * tokens_per_frame + y * W + x
                        if idx < s:
                            query_indices.append(idx)

        query_indices = torch.tensor(query_indices, device=q.device)

        # 3. Compute Attention Scores (Raw)
        # [B, S, N, D] -> [B, N, S, D]
        q_head = q.transpose(1, 2)
        k_head = k.transpose(1, 2)

        # Select Q: [B, N, Q_sel, D]
        q_selected = q_head[:, :, query_indices, :]

        # Dot Product: [B, N, Q_sel, S_all]
        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(q_selected, k_head.transpose(-2, -1)) * scale

        # 4. Generate Mask based on crossview_attn_type
        key_indices = torch.arange(seq_len, device=q.device)

        V_q = query_indices // tokens_per_view
        F_q = (query_indices % tokens_per_view) // tokens_per_frame

        V_k = key_indices // tokens_per_view
        F_k = (key_indices % tokens_per_view) // tokens_per_frame

        V_q_exp = V_q.unsqueeze(1)
        F_q_exp = F_q.unsqueeze(1)
        V_k_exp = V_k.unsqueeze(0)
        F_k_exp = F_k.unsqueeze(0)

        mask = torch.ones((query_indices.shape[0], seq_len), device=q.device, dtype=torch.bool)

        if "view-wise-attn" in crossview_attn_type:
            conn_matrix = torch.zeros((num_views, num_views), device=q.device, dtype=torch.bool)
            for src, tgts in mv_map.items():
                if src < num_views:
                    valid_targets = [t for t in tgts if t < num_views]
                    conn_matrix[src, valid_targets] = True
            mask = conn_matrix[V_q_exp, V_k_exp]

        elif "spatial-crossview-attn-full-singleview-attn" in crossview_attn_type:
            same_view_mask = (V_q_exp == V_k_exp)
            same_frame_mask = (F_q_exp == F_k_exp)
            mask = same_view_mask | same_frame_mask

        # 5. Apply Mask and Softmax
        mask_broad = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~mask_broad, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 6. Save
        os.makedirs(save_dir, exist_ok=True)
        torch.save(attn_weights.cpu(), os.path.join(save_dir, f"attn_weights_layer_{layer_idx}.pt"))

        meta_path = os.path.join(os.path.dirname(save_dir), "query_metadata.pt")
        if not os.path.exists(meta_path):
             torch.save({
                 "num_views": num_views,
                 "grid_sizes": grid_sizes[0].cpu(),
                 "query_indices": query_indices.cpu(),
                 "target_frames": target_frames,
                 "h_points": h_points,
                 "w_points": w_points,
                 "tokens_per_view": tokens_per_view,
                 "tokens_per_frame": tokens_per_frame
             }, meta_path)

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        pos_emb_fn,
        dtype,
        num_views=1,
        cross_view_flex_attn=None,
        crossview_attn_type="full",
        cond_mask=None,
        kv_cache=None,
        return_kv=False,
        score_vis_dir=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # pre-compute small ProPE branch qkv from input features (before x is overwritten)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()


        proj_mats = None
        if kv_cache is not None:
            # TODO: kv_cache支持6view
            k_cache = kv_cache[0].to(x.device)
            v_cache = kv_cache[1].to(x.device)
            assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, b]
            if k_cache.shape[0] == 1:
                k_cache = k_cache.repeat(b, 1, 1, 1)
                v_cache = v_cache.repeat(b, 1, 1, 1)
            k = torch.cat([k_cache, k], dim=1).contiguous()
            v = torch.cat([v_cache, v], dim=1).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=1).contiguous()
            grid_sizes = grid_sizes.clone()
            grid_sizes[:, 0] = grid_sizes[:, 0] + k_cache.shape[1] // (grid_sizes[0,1] * grid_sizes[0,2])

            q_padding, k = pos_emb_fn(
                q=q_padding,
                k=k,
                grid_sizes=grid_sizes,
            )
            q = q_padding[:, -s:].contiguous()
            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)
        else:
            q = rearrange(q, 'b (nv s) n d -> (b nv) s n d', nv=num_views)
            k = rearrange(k, 'b (nv s) n d -> (b nv) s n d', nv=num_views)

            tmp_grid_sizes = grid_sizes.clone()
            tmp_grid_sizes = tmp_grid_sizes.repeat_interleave(num_views, dim=0)
            tmp_grid_sizes[:, 0] = tmp_grid_sizes[:, 0] // num_views

            # --- START MODIFICATION ---
            grid_offsets = None
            if num_views > 1:
                b_nv = q.shape[0]
                grid_offsets = torch.zeros((b_nv, 2), dtype=torch.long, device=q.device)

                H = grid_sizes[0, 1].item()
                W = grid_sizes[0, 2].item()

                view_indices = torch.arange(num_views, device=q.device).repeat(b_nv // num_views)

                grid_offsets[:, 0] = view_indices * H  # Height Offset
                grid_offsets[:, 1] = view_indices * W  # Width Offset

            q, k = pos_emb_fn(
                q=q,
                k=k,
                grid_sizes=tmp_grid_sizes,
                grid_offsets=grid_offsets,
            )
            # --- END MODIFICATION ---
            q = rearrange(q, '(b nv) s n d -> b (nv s) n d', nv=num_views)
            k = rearrange(k, '(b nv) s n d -> b (nv s) n d', nv=num_views)
            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)

        if envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            gate_compress = self.to_gate_compress(x).view(b, s, n, d).to(dtype)
            if dist.is_initialized() and dist.get_rank() == 0:
                if torch.isnan(gate_compress).any():
                    print(f"[Error] gate_compress has nan values.")
                    if torch.isnan(x).any():
                        print(f"[Error] x has nan values.")
                    if torch.isnan(self.to_gate_compress.weight).any():
                        print(f"[Error] to_gate_compress weight has nan values.")
                    if torch.isnan(self.to_gate_compress.bias).any():
                        print(f"[Error] to_gate_compress bias has nan values.")

        # [INSERT VISUALIZATION LOGIC HERE]
        if score_vis_dir is not None:
            try:
                layer_idx = int(self.prefix.split('.')[-2])
            except:
                layer_idx = -1
                parts = self.prefix.split('.')
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)

            if layer_idx % 3 == 0:
                self.save_attention_map(q.clone(), k.clone(), grid_sizes, num_views, score_vis_dir, layer_idx, crossview_attn_type=crossview_attn_type)

        if num_views > 1:
            x = _multiview_attention(
                q, k, v, crossview_attn_type, num_views, grid_sizes,
                seq_lens, self.window_size, cross_view_flex_attn, cond_mask)
        else:
            attn_kwargs = {}
            attn_mask = _build_causal_mask(crossview_attn_type, grid_sizes, x.device)
            if attn_mask is not None:
                attn_kwargs["attn_mask"] = attn_mask

            if cond_mask is not None and kv_cache is None:
                x = _singleview_attention_with_cond_mask(
                    q, k, v, seq_lens, self.window_size,
                    cond_mask, grid_sizes, attn_kwargs)
            else:
                if not USE_PPU and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
                    x, _ = self.attn1(q, k, v, gate_compress=gate_compress)
                else:
                    x = attention(
                        q=q, k=k, v=v,
                        k_lens=seq_lens,
                        window_size=self.window_size,
                        **attn_kwargs,
                    )

        x = x.to(dtype)

        # output
        x = x.flatten(2)  # shape: [B, L, C]
        x = self.o(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x


# ---------- Cross Attention ----------

class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype, grid_sizes, num_views=1, cond_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        if cond_mask is not None:
            if num_views == 1:
                # compute query, key, value
                q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
                k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
                v = self.v(context.to(dtype)).view(b, -1, n, d)

                b, s, _ = x.shape
                n_frames = cond_mask.shape[1]
                n_tokens_per_img = s // n_frames
                q = rearrange(q, 'b (f s) ... -> b f s ...', f=n_frames, s=n_tokens_per_img)

                noise_seq_idxs = (~cond_mask[0]).nonzero(as_tuple=False).squeeze(-1)
                q_noise = q[:, noise_seq_idxs].contiguous().view(b, -1, n, d)
                x_noncond = attention(
                    q=q_noise,
                    k=k,
                    v=v,
                    k_lens=context_lens,
                )
                x = torch.zeros((b, n_frames, n_tokens_per_img, n, d), device=x_noncond.device, dtype=x_noncond.dtype)
                x[:, noise_seq_idxs] = x_noncond.view(b, noise_seq_idxs.shape[0], n_tokens_per_img, n, d)
                x = x.view(b, -1, n, d).contiguous()
                x = x.to(dtype)

                x = x.flatten(2)
                x = self.o(x)

            else:
                assert not USE_PPU, "Block sparse attention is only implemented for non-PPU."
                vf, h, w = grid_sizes[0].tolist()
                num_frames_each_view = vf // num_views
                num_tokens_each_frame = h * w
                num_tokens_text_prompt_each_view = context.shape[1] // num_views

                block_mask = cross_view_text_block_mask(
                    num_views = num_views,
                    num_frames_each_view = num_frames_each_view,
                    num_tokens_each_frame = num_tokens_each_frame,
                    num_tokens_text_prompt_each_view = num_tokens_text_prompt_each_view,
                    device_str = str(x.device),
                    cond_mask = cond_mask[0] if cond_mask is not None else None,
                )

                mask = cond_mask[0].repeat_interleave(num_tokens_each_frame)
                x = x[:, ~mask].contiguous()

                q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
                k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
                v = self.v(context.to(dtype)).view(b, -1, n, d)
                x = block_sparse_attention(
                    q=q,
                    k=k,
                    v=v,
                    block_mask=block_mask,
                )

                x = x.to(dtype)
                x = x.flatten(2)
                x_noise = self.o(x)

                x = torch.zeros((b, vf * num_tokens_each_frame, n * d), device=x_noise.device, dtype=x_noise.dtype)
                x[:, ~mask] = x_noise

        else:
            q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
            k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
            v = self.v(context.to(dtype)).view(b, -1, n, d)
            x = attention(
                q.to(dtype),
                k.to(dtype),
                v.to(dtype),
                k_lens=context_lens
            )
            x = x.to(dtype)

            x = x.flatten(2)
            x = self.o(x)

        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype),
            k_img.to(dtype),
            v_img.to(dtype),
            k_lens=None
        )
        img_x = img_x.to(dtype)
        x = attention(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            k_lens=context_lens
        )
        x = x.to(dtype)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}
