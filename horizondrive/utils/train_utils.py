import torch
import torch.nn.functional as F
import gc
from decord import VideoReader
from contextlib import contextmanager
from PIL import Image
import logging
import torchvision.transforms.functional as TF
from typing import Optional, Tuple
import numpy as np
from einops import rearrange

logger = logging.getLogger(__name__)

WAN_FUN_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"






    
@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

    
### input preprocess utils for training and inference ###

def resize_mask(mask: torch.Tensor, latent: Optional[torch.Tensor], process_first_frame_only: bool = True, latent_size: Optional[tuple] = None) -> torch.Tensor:
    """Resize a binary mask (B, C, T, H, W) to match latent (B, C, T', H', W').

    - If process_first_frame_only, upscale the first frame to T'=1 and the rest to T'-1, then concat.
    - Accept latent=None when latent_size is provided explicitly.
    """
    if latent_size is None:
        if latent is None:
            raise ValueError("Either latent or latent_size must be provided")
        latent_size = latent.size()

    if mask.dim() != 5:
        raise ValueError(f"mask must be 5D (B,C,T,H,W), got {mask.shape}")

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = max(target_size[0] - 1, 0)
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
    return resized_mask

def batch_encode_vae(
    tensor: torch.Tensor,
    vae,
    num_cameras: int,
    mini_batch: int =1,
) -> torch.Tensor:
    """Encode (B, N*F, C, H, W) to latents (B, C, N*F, H//s, W//s) with VAE in mini-batches.

    - The input should be laid out as (B, N*F, C, H, W).
    - This matches training code path and avoids OOM by chunking on (B*N).
    """
    pixel_values = rearrange(tensor, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    new_pixel_values = []
    for i in range(0, pixel_values.shape[0], mini_batch):
        pixel_values_bs = pixel_values[i : i + mini_batch]
        pixel_values_bs = vae.encode(pixel_values_bs)[0]
        pixel_values_bs = pixel_values_bs.sample()
        new_pixel_values.append(pixel_values_bs)
    new_pixel_values = torch.cat(new_pixel_values, dim=0)
    new_pixel_values = rearrange(new_pixel_values, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    return new_pixel_values



def prepare_clip_context(
    clip_image_encoder,
    clip_pixel_values: Optional[torch.FloatTensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """
    Build clip_context for training:
      - clip_pixel_values: iterable of single images (H,W,3) as tensors on CPU
      - Returns concatenated contexts of shape (B, C_ctx, T)
    """
    device = device or clip_image_encoder.device
    dtype = dtype or clip_image_encoder.dtype
    if clip_pixel_values is not None:
        clip_image = Image.fromarray(np.uint8(clip_pixel_values.float().cpu().numpy()))
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, dtype)
        clip_context = clip_image_encoder([clip_image[:, None, :, :]])
    else:
        # fallback to black image, then zeros_like
        black = Image.new("RGB", (512, 512), color=(0, 0, 0))
        img_t = TF.to_tensor(black).sub_(0.5).div_(0.5).to(device, dtype)
        ctx = clip_image_encoder([img_t[:, None, :, :]])
        clip_context = torch.zeros_like(ctx)
    
    return clip_context

def _repeat_first_frame_concat_rest(x: torch.Tensor, repeat_times: int = 4) -> torch.Tensor:
    """Given (B, C, F, H, W), make time F' = F+3 by repeating the first frame 4 times then concatenating the rest."""
    return torch.concat(
        [
            torch.repeat_interleave(x[:, :, 0:1], repeats=repeat_times, dim=2),
            x[:, :, 1:],
        ],
        dim=2,
    )


def prepare_mask_condition(
    mask: torch.Tensor,
    latents: torch.Tensor,
    num_cameras: int,
    temporal_compression_ratio: int = 4,
) -> torch.Tensor:
    """Convert raw binary mask (B, N*F, C, H, W) to resized latent mask (B, C, N*F, H', W').

    Steps follow the training/inference logic:
    - pad by repeating first frame 4x, then group every 4 frames
    - resize to match latent temporal/spatial shape
    """
    mask = rearrange(mask, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    mask = _repeat_first_frame_concat_rest(mask, repeat_times=temporal_compression_ratio).contiguous()
    mask = mask.view(
        mask.shape[0], mask.shape[2] // temporal_compression_ratio, temporal_compression_ratio, mask.shape[3], mask.shape[4]
    )
    mask = mask.transpose(1, 2)  # (B*N, 4, F', H, W)

    b, c, nf, h, w = latents.size()
    n = num_cameras
    mask_condition = resize_mask(1 - mask, latent=None, latent_size=(b * n, c, nf // n, h, w), process_first_frame_only=False)
    mask_condition = rearrange(mask_condition, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
    
    return mask_condition


def align_first_segment_latents(
    video: torch.Tensor,
    vae,
    num_cameras: int,
    t_compression_ratio: int = 4,
    mini_batch: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode video with optional first-frame padding for rollout latent alignment."""
    # Pad first frame 4 times so the first segment's latent distribution aligns
    # with subsequent segments (which always have preceding context frames).
    padding_first_frame_n = 4
    gt_latents = batch_encode_vae(video, vae, num_cameras=num_cameras, mini_batch=mini_batch)

    video_per_cam = rearrange(video, "b (n f) c h w -> (b n) c f h w", n=num_cameras)
    first_frame = video_per_cam[:, :, 0:1]
    pad_frames = first_frame.repeat(1, 1, padding_first_frame_n, 1, 1)
    video_padded = torch.cat([pad_frames, video_per_cam], dim=2)
    video_for_encode = rearrange(video_padded, "(b n) c f h w -> b (n f) c h w", n=num_cameras)
    latents_padded = batch_encode_vae(video_for_encode, vae, num_cameras=num_cameras, mini_batch=mini_batch)

    lat_f_per_cam = latents_padded.shape[2] // num_cameras
    pad_lat_f = (padding_first_frame_n + t_compression_ratio - 1) // t_compression_ratio
    aligned_latents = torch.cat(
        [
            latents_padded[:, :, i * lat_f_per_cam + pad_lat_f : (i + 1) * lat_f_per_cam]
            for i in range(num_cameras)
        ],
        dim=2,
    )
    return gt_latents, aligned_latents


def replace_condition_with_gt_latents(
    final_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    num_cameras: int,
    num_condition_latent_frames: int,
) -> torch.Tensor:
    """Replace rollout condition latents with directly encoded GT latents before decode."""
    if num_condition_latent_frames <= 0:
        return final_latents

    gt_f_per_cam = gt_latents.shape[2] // num_cameras
    out_f_per_cam = final_latents.shape[2] // num_cameras
    cond_f = min(num_condition_latent_frames, gt_f_per_cam)
    result = final_latents.clone()
    for i in range(num_cameras):
        out_sl = slice(i * out_f_per_cam, i * out_f_per_cam + cond_f)
        gt_sl = slice(i * gt_f_per_cam, i * gt_f_per_cam + cond_f)
        result[:, :, out_sl] = gt_latents[:, :, gt_sl]
    return result






def prepare_action_condition(actions: torch.Tensor, num_cameras: int=1, temporal_compression_ratio: int=4) -> torch.Tensor:
    """action F to diffusion F'
    [B, NF, C] -> [B, NF', C]
    """
    a = rearrange(actions, "b (n f) c -> (b n) f c", n=num_cameras)
    a = torch.concat(
        [
            torch.repeat_interleave(a[:, 0:1], repeats=temporal_compression_ratio, dim=1),
            a[:, 1:],
        ],
        dim=1,
    )
    a = a.view(a.shape[0], a.shape[1] // temporal_compression_ratio, temporal_compression_ratio, a.shape[2])
    a = a.mean(dim=2)
    a = rearrange(a, "(b n) f c -> b (n f) c", n=num_cameras)
    return a



