import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    init_distributed_environment, initialize_model_parallel,
                    xFuserLongContextAttention)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
        dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output)

def _rope_apply_rank_slice(x, grid_sizes, pos_emb_fn, rank):
    """Apply RoPE to the local sequence-parallel slice."""
    if not hasattr(pos_emb_fn, "freqs"):
        raise ValueError("sequence-parallel Wan attention requires a RoPE position embedding")

    b, s, n, d = x.shape
    c = d // 2
    freqs = pos_emb_fn.freqs.to(x.device)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    global_start = rank * s
    positions = torch.arange(global_start, global_start + s, device=x.device)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        frame_area = h * w
        valid = positions < seq_len
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        t_idx = safe_positions // frame_area
        rem = safe_positions % frame_area
        h_idx = rem // w
        w_idx = rem % w

        freqs_i = torch.cat([freqs[0][t_idx], freqs[1][h_idx], freqs[2][w_idx]], dim=-1).unsqueeze(1)
        freqs_i = torch.where(valid.view(s, 1, 1), freqs_i, torch.ones_like(freqs_i))

        x_i = torch.view_as_complex(x[i].to(torch.float32).reshape(s, n, -1, 2))
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(x_i)
    return torch.stack(output).float()


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     pos_emb_fn,
                     dtype=torch.bfloat16,
                     num_views=1,
                     cross_view_flex_attn=None,
                     crossview_attn_type="full",
                     cond_mask=None,
                     kv_cache=None,
                     return_kv=False,
                     score_vis_dir=None):
    if num_views != 1:
        raise NotImplementedError("sequence-parallel Wan attention currently supports num_views=1 only")
    if kv_cache is not None or return_kv:
        raise NotImplementedError("sequence-parallel Wan attention does not support kv_cache yet")

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    sp_rank = get_sequence_parallel_rank()
    q = _rope_apply_rank_slice(q, grid_sizes, pos_emb_fn, sp_rank)
    k = _rope_apply_rank_slice(k, grid_sizes, pos_emb_fn, sp_rank)

    if xFuserLongContextAttention is None:
        world_size = get_sequence_parallel_world_size()
        k_parts = [torch.empty_like(k) for _ in range(world_size)]
        v_parts = [torch.empty_like(v) for _ in range(world_size)]
        dist.all_gather(k_parts, k.contiguous())
        dist.all_gather(v_parts, v.contiguous())
        k = torch.cat(k_parts, dim=1)
        v = torch.cat(v_parts, dim=1)
        key_valid = torch.arange(k.size(1), device=k.device).unsqueeze(0) < seq_lens.to(k.device).unsqueeze(1)
        x = F.scaled_dot_product_attention(
            half(q).transpose(1, 2),
            half(k).transpose(1, 2),
            half(v).transpose(1, 2),
            attn_mask=key_valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()
    else:
        x = xFuserLongContextAttention()(
            None,
            query=half(q),
            key=half(k),
            value=half(v),
            window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
