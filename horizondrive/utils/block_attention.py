import torch
from functools import lru_cache
from block_sparse_attn import block_sparse_attn_func

BLOCK_SIZE = 128
mv_map = {
    0: [0, 1, 2],
    1: [0, 1, 5], 
    2: [0, 2, 3],
    3: [2, 3, 4],
    4: [3, 4, 5],
    5: [1, 4, 5],
}


def create_view_connectivity_matrix(num_views, device):
    """创建视图连接矩阵 - 向量化实现"""
    conn_matrix = torch.zeros((num_views, num_views), device=device, dtype=torch.bool)
    for view_id, target_views in mv_map.items():
        if view_id < num_views:
            # 只设置允许连接的视图
            valid_targets = [v for v in target_views if v < num_views]
            if valid_targets:
                conn_matrix[view_id, valid_targets] = True
    return conn_matrix


def block_mask_with_cond_mask(base_block_mask, cond_mask, num_tokens_each_frame):
    """根据条件掩码调整基础块掩码 - 向量化实现"""
    if cond_mask is None:
        return base_block_mask
    
    num_blocks_each_frame = num_tokens_each_frame // BLOCK_SIZE
    cond_frame_mask = ~(cond_mask.unsqueeze(1) & (~cond_mask).unsqueeze(0))
    cond_block_mask = cond_frame_mask.repeat_interleave(num_blocks_each_frame, dim=0)
    cond_block_mask = cond_block_mask.repeat_interleave(num_blocks_each_frame, dim=1)
    adjusted_block_mask = base_block_mask & cond_block_mask.to(base_block_mask.device)
    
    return adjusted_block_mask


@lru_cache(maxsize=4)
def _multi_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str):
    """优化后的多视图全注意力块掩码 - 完全向量化"""
    device = torch.device(device_str)
    num_blocks_each_view = num_frames_each_view * num_tokens_each_frame // BLOCK_SIZE
    
    # 向量化创建基础掩码
    view_conn = create_view_connectivity_matrix(num_views, device)
    
    # 使用repeat_interleave进行批量扩展
    block_mask = view_conn.repeat_interleave(num_blocks_each_view, dim=0)
    block_mask = block_mask.repeat_interleave(num_blocks_each_view, dim=1)
    
    return block_mask


def multi_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str, cond_mask=None):
    block_mask = _multi_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask = block_mask_with_cond_mask(block_mask, cond_mask, num_tokens_each_frame)
    
    return block_mask


@lru_cache(maxsize=4)
def _cross_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str):
    """优化后的跨视图空间注意力块掩码 - 完全向量化"""
    device = torch.device(device_str)
    num_blocks_each_frame = num_tokens_each_frame // BLOCK_SIZE
    num_blocks = num_blocks_each_frame * num_frames_each_view * num_views

    view_conn = create_view_connectivity_matrix(num_views, device)
    
    # 生成帧对角线掩码 [num_frames_each_view, num_frames_each_view]
    frame_diag_mask = torch.eye(num_frames_each_view, dtype=torch.bool, device=device)
    
    # 扩展为视角-帧掩码 [num_views, num_frames_each_view, num_views, num_frames_each_view]
    view_frame_mask = view_conn.unsqueeze(1).unsqueeze(3) & frame_diag_mask.unsqueeze(0).unsqueeze(2)
    
    block_mask = view_frame_mask.repeat_interleave(num_blocks_each_frame, dim=1)
    block_mask = block_mask.repeat_interleave(num_blocks_each_frame, dim=3)
    
    block_mask = block_mask.reshape(num_blocks, num_blocks)
    
    return block_mask


def cross_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str, cond_mask=None):
    block_mask = _cross_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask = block_mask_with_cond_mask(block_mask, cond_mask, num_tokens_each_frame)
    return block_mask


@lru_cache(maxsize=4)
def _single_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str):
    """优化后的单视图全注意力块掩码 - 完全向量化"""
    device = torch.device(device_str)
    num_blocks_each_view = num_frames_each_view * num_tokens_each_frame // BLOCK_SIZE

    view_conn = torch.eye(num_views, device=device, dtype=torch.bool)
    # 使用repeat_interleave进行批量扩展
    block_mask = view_conn.repeat_interleave(num_blocks_each_view, dim=0)
    block_mask = block_mask.repeat_interleave(num_blocks_each_view, dim=1)

    return block_mask


def single_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str, cond_mask=None):
    block_mask = _single_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask = block_mask_with_cond_mask(block_mask, cond_mask, num_tokens_each_frame)
    return block_mask


def _all_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str):
    # 不同view的同一时刻的帧之间全部做attention，不再考虑mv_map
    device = torch.device(device_str)
    num_blocks_each_frame = num_tokens_each_frame // BLOCK_SIZE
    num_blocks = num_blocks_each_frame * num_frames_each_view * num_views
    
    # 创建时间步对角线掩码 [num_frames_each_view, num_frames_each_view]
    # 确保同一时刻的帧可以相互关注
    time_diag_mask = torch.eye(num_frames_each_view, dtype=torch.bool, device=device)
    
    # 创建全连接的视图掩码 [num_views, num_views]
    # 所有视图之间都建立连接（全1矩阵）
    view_conn = torch.ones(num_views, num_views, dtype=torch.bool, device=device)
    
    # 组合视图和时间掩码 [num_views, num_frames_each_view, num_views, num_frames_each_view]
    # 只有同一时刻的不同视图帧之间才建立连接
    view_time_mask = view_conn.unsqueeze(1).unsqueeze(3) & time_diag_mask.unsqueeze(0).unsqueeze(2)
    
    # 扩展到块级别
    block_mask = view_time_mask.repeat_interleave(num_blocks_each_frame, dim=1)
    block_mask = block_mask.repeat_interleave(num_blocks_each_frame, dim=3)
    
    # 重塑为最终的块掩码矩阵
    block_mask = block_mask.reshape(num_blocks, num_blocks)
    
    return block_mask
    


@lru_cache(maxsize=4)
def _cross_view_spatial_attn_full_singleview_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str):
    # block_mask_spatial = cross_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask_spatial = _all_view_spatial_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask_singleview = _single_view_full_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    combined_mask = block_mask_spatial | block_mask_singleview
    return combined_mask


def cross_view_spatial_attn_full_singleview_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str, cond_mask=None):
    block_mask = _cross_view_spatial_attn_full_singleview_attn_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, device_str)
    block_mask = block_mask_with_cond_mask(block_mask, cond_mask, num_tokens_each_frame)
    return block_mask



# def cross_view_text_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, num_tokens_text_prompt, device_str, cond_mask=None):
#     # q的长度是 (num_views * num_frames_each_view * num_tokens_each_frame)； k,v的长度是 num_tokens_text_prompt = num_views * text_prompt_length_each_view
#     # 做cross-attn，q只和当前view的text_prompt做attention
#     device = torch.device(device_str)
#     block_mask = torch.eye(num_views, device=device, dtype=torch.bool)
#     block_mask = block_mask.repeat_interleave(num_frames_each_view, dim=0)
#     if cond_mask is not None:
#         block_mask = block_mask & (~cond_mask).unsqueeze(1).repeat(1, num_views)

#     num_blocks_each_frame = num_tokens_each_frame // BLOCK_SIZE
#     block_mask = block_mask.repeat_interleave(num_blocks_each_frame, dim=0)
#     num_blocks_text_prompt_each_view = num_tokens_text_prompt // BLOCK_SIZE // num_views
#     block_mask = block_mask.repeat_interleave(num_blocks_text_prompt_each_view, dim=1)
#     return block_mask


def cross_view_text_block_mask(num_views, num_frames_each_view, num_tokens_each_frame, num_tokens_text_prompt_each_view, device_str, cond_mask=None):
    # q的长度是 (num_views * num_frames_each_view * num_tokens_each_frame)； k,v的长度是 num_tokens_text_prompt = num_views * text_prompt_length_each_view
    # 做cross-attn，q只和当前view的text_prompt做attention
    device = torch.device(device_str)
    block_mask = torch.eye(num_views, device=device, dtype=torch.bool)
    block_mask = block_mask.repeat_interleave(num_frames_each_view, dim=0)
    
    if cond_mask is not None:
        # 根据cond_mask过滤掉不需要做attention的q token
        # 只保留cond_mask为False的位置（即需要做attention的q token）
        non_cond_indices = torch.where(~cond_mask)[0]
        block_mask = block_mask[non_cond_indices]  # 只保留需要做attention的q token对应的行
    
    num_blocks_each_frame = num_tokens_each_frame // BLOCK_SIZE
    block_mask = block_mask.repeat_interleave(num_blocks_each_frame, dim=0)
    num_blocks_text_prompt_each_view = num_tokens_text_prompt_each_view // BLOCK_SIZE
    block_mask = block_mask.repeat_interleave(num_blocks_text_prompt_each_view, dim=1)
    return block_mask



def align_to_block_size(func):
    def wrapper(q, k, v, *args, **kwargs):
        batch_size, seqlen, nheads, headdim = q.shape
        padded_seqlen = seqlen
        kv_seqlen = k.shape[1]
        padded_kv_seqlen = kv_seqlen

        if (seqlen % BLOCK_SIZE != 0):
            pad_len = BLOCK_SIZE - (seqlen % BLOCK_SIZE)
            kv_pad_len = BLOCK_SIZE - (kv_seqlen % BLOCK_SIZE)
            q = torch.cat([q, torch.zeros(batch_size, pad_len, nheads, headdim, device=q.device, dtype=q.dtype)], dim=2)
            k = torch.cat([k, torch.zeros(batch_size, kv_pad_len, nheads, headdim, device=k.device, dtype=k.dtype)], dim=2)
            v = torch.cat([v, torch.zeros(batch_size, kv_pad_len, nheads, headdim, device=v.device, dtype=v.dtype)], dim=2)
            padded_seqlen = seqlen + pad_len
            padded_kv_seqlen = kv_seqlen + kv_pad_len

        q = q.reshape(batch_size, padded_seqlen, nheads, headdim)
        k = k.reshape(batch_size, padded_kv_seqlen, nheads, headdim)
        v = v.reshape(batch_size, padded_kv_seqlen, nheads, headdim)

        output = func(q, k, v, *args, **kwargs)
        output = output.view(batch_size, padded_seqlen, nheads, headdim)
        return output[:, :seqlen, :, :].contiguous()
    return wrapper


@align_to_block_size
def block_sparse_attention(q, k, v, block_mask=None, dropout_p=0.0, is_causal=False, exact_streaming=False):
    """
    q, k, v: (batch_size, seq_len, nheads, headdim)
    dropout_p: float
    is_causal: bool
    exact_streaming: bool
    
    return: (batch_size, seq_len, nheads, headdim)
    
    """
    
    device = q.device
    batch_size, seqlen, nheads, headdim = q.shape
    kv_seqlen = k.shape[1]
    assert seqlen % BLOCK_SIZE == 0, f"seqlen: {seqlen} is not divisible by {BLOCK_SIZE}"
    q = q.reshape(batch_size * seqlen, nheads, headdim)
    k = k.reshape(batch_size * kv_seqlen, nheads, headdim)
    v = v.reshape(batch_size * kv_seqlen, nheads, headdim)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.arange(0, (batch_size + 1) * kv_seqlen, step=kv_seqlen, dtype=torch.int32, device=device)
    head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)

    assert block_mask is not None, "block_mask is required for block sparse attention"

    # 如果block_mask只有二维，则扩展为四维
    if block_mask.dim() == 2:
        block_mask = block_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1).contiguous()

    # output: (batch_size * seqlen, nheads, headdim)
    output = block_sparse_attn_func(q, k, v, cu_seqlens, cu_kv_seqlens, head_mask_type, None, block_mask, seqlen, kv_seqlen, dropout_p, is_causal=is_causal, exact_streaming=exact_streaming)
    
    return output.view(batch_size, seqlen, nheads, headdim).contiguous()
