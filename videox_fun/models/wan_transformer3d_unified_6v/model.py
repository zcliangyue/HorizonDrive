# Main UnifiedTransformer3DModel for Wan Transformer
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import functools
import glob
import json
import math
import os
from typing import Any, Dict

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from einops import rearrange
from accelerate.logging import get_logger

from ..cache_utils import TeaCache, cfg_skip, disable_cfg_skip, enable_cfg_skip
from videox_fun.utils.utils import construct_emb_cls

from .embeddings import (
    sinusoidal_embedding_batchwise,
    rope_params,
    get_1d_rotary_pos_embed_riflex,
    RopeEmb,
    ResizeRopeEmb,
)
from .conditioning import (
    MLPProj,
    ViewIDEmbedding,
    LearnbleViewIDEmb,
)
from .attention import CrossViewMask, fused_flex_attention
from .blocks import WanAttentionBlock, Head

LOG_NAME = "trainer"
LOG_LEVEL = "INFO"
logger = get_logger(LOG_NAME, LOG_LEVEL)


class UnifiedTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        in_dim_control_adapter=24,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        position_embedding_kwargs={},
        additional_condition_kwargs={},
        use_view_embedding=True,
        prefix="Wan",
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            additional_condition_kwargs (`dict`, *optional*):
                Additional kwargs for conditional inputs
        """

        super().__init__()
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        for condition_name, condition_args in additional_condition_kwargs.items():
            kwargs = {
                'dim': dim,
                'patch_size': patch_size,
            }
            if "action" in condition_name:
                kwargs["freq_dim"] = freq_dim
            emb_cls = construct_emb_cls(condition_args, kwargs)
            setattr(self, condition_name, emb_cls)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, prefix=f"{prefix}.blocks.{i}")
            for i in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.pos_emb_fn = construct_emb_cls(
            position_embedding_kwargs,
            {
                'head_dim': d,
                'max_seq_len': 1024
            }
        )

        self.view_embedding = None
        if use_view_embedding:
            self.view_embedding = ViewIDEmbedding(freq_dim=freq_dim, dim=dim)

        self._view_ids = None

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(
                    in_dim_ref_conv,
                    dim,
                    kernel_size=patch_size[1:],
                    stride=patch_size[1:]
                )
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True
    ):
        self.teacache = TeaCache(
            coefficients,
            num_steps,
            rel_l1_thresh=rel_l1_thresh,
            num_skip_start_steps=num_skip_start_steps,
            offload=offload
        )

    def disable_teacache(self):
        self.teacache = None

    @enable_cfg_skip()
    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    @disable_cfg_skip()
    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    ALL_CAMERA_NAMES = [
        "camera_front", "camera_front_left", "camera_front_right",
        "camera_rear_right", "camera_rear", "camera_rear_left",
    ]

    def set_view_ids(self, camera_names):
        view_ids = [self.ALL_CAMERA_NAMES.index(name) for name in camera_names]
        self._view_ids = torch.tensor(view_ids, dtype=torch.long)

    def get_view_ids(self):
        return self._view_ids

    def get_cross_view_flex_attn(self, seq_lens, grid_sizes, num_views, device):
        current_config= {
            'seq_len': seq_lens[0].item(),
        }
        if hasattr(self, 'cross_view_flex_attn') and hasattr(self, 'current_config') and self.current_config == current_config:
            return self.cross_view_flex_attn
        else:
            block_mask_fn = CrossViewMask()
            block_mask = block_mask_fn.get_block_mask(
                num_views, seq_lens, grid_sizes, device)
            self.cross_view_flex_attn = functools.partial(fused_flex_attention, block_mask=block_mask)
            self.current_config = current_config
        return self.cross_view_flex_attn

    def _run_blocks(self, x, e0, seq_lens, grid_sizes, context, context_lens,
                    dtype, num_views, cross_view_flex_attn, crossview_attn_type,
                    cond_mask, kv_cache_dict, return_kv, skip_cross_attn,
                    score_vis_dir, offload_kv_cache):
        """Run all transformer blocks. Shared by both TeaCache and non-TeaCache paths."""
        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.pos_emb_fn,
                    context,
                    context_lens,
                    dtype,
                    num_views,
                    cross_view_flex_attn,
                    crossview_attn_type,
                    cond_mask,
                    kv_cache_dict.get(i, None),
                    return_kv,
                    skip_cross_attn,
                    score_vis_dir,
                    **ckpt_kwargs,
                )
            else:
                kwargs = dict(
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    pos_emb_fn=self.pos_emb_fn,
                    context=context,
                    context_lens=context_lens,
                    dtype=dtype,
                    num_views=num_views,
                    cross_view_flex_attn=cross_view_flex_attn,
                    crossview_attn_type=crossview_attn_type,
                    cond_mask=cond_mask,
                    kv_cache=kv_cache_dict.get(i, None),
                    return_kv=return_kv,
                    skip_cross_attn=skip_cross_attn,
                    score_vis_dir=score_vis_dir
                )
                x = block(x, **kwargs)

            if return_kv:
                x, kv_cache = x
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())

        return x, kv_cache_dict_ret

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context=None,
        seq_len=None,
        clip_fea=None,
        y=None,
        full_ref=None,
        cond_flag=True,
        num_views=1,
        dtype=None,
        crossview_attn_type="full",
        step=None,
        additional_conditions={},
        cond_mask=None,
        kv_cache_dict={},
        return_kv=False,
        skip_cross_attn=False,
        offload_kv_cache=False,
        score_vis_dir=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if len(t.shape) != 1 and len(t.shape) != 2:
            raise ValueError(f"t must be of shape [B] or [B, F], but got {t.shape}")

        # params
        device = self.patch_embedding.weight.device
        if dtype is None:
            dtype = x.dtype

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]

        for condition_name, y_add in additional_conditions.items():
            if condition_name == "action":
                continue
            if hasattr(self, condition_name):
                emb_fn = getattr(self, condition_name)
                x = emb_fn(x, y_add)
            else:
                raise ValueError(f"condition {condition_name} not found in the model.")

        if self.view_embedding is not None and num_views > 1:
            if self._view_ids is not None:
                view_ids = self._view_ids.to(device)
            else:
                view_ids = torch.arange(num_views, device=device)
            x = self.view_embedding(x, view_ids, num_views=num_views, device=device, dtype=dtype)

        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with torch.amp.autocast('cuda',dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_batchwise(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
            if "action" in additional_conditions and hasattr(self, "action"):
                if len(e0.shape) == 3:
                    e0 = e0.unsqueeze(1).repeat(1, grid_sizes[0][0], 1, 1)
                a0 = self.action(additional_conditions["action"])
                e0 = e0 + a0

        # context
        context_lens = None
        if context is not None:
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

            context = rearrange(context, '(b v) l c -> b (v l) c', v=num_views)

            if clip_fea is not None:
                if num_views > 1:  # TODO
                    raise NotImplementedError("clip_fea with num_views > 1 is not implemented yet.")
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)

        # block_mask
        if num_views > 1 and crossview_attn_type == 'flex':
            cross_view_flex_attn = self.get_cross_view_flex_attn(
                seq_lens, grid_sizes, num_views, device)
        else:
            cross_view_flex_attn = None

        # TeaCache
        kv_cache_dict_ret = {}
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc

        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()
                x, kv_cache_dict_ret = self._run_blocks(
                    x, e0, seq_lens, grid_sizes, context, context_lens,
                    dtype, num_views, cross_view_flex_attn, crossview_attn_type,
                    cond_mask, kv_cache_dict, return_kv, skip_cross_attn,
                    score_vis_dir, offload_kv_cache)

                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            x, kv_cache_dict_ret = self._run_blocks(
                x, e0, seq_lens, grid_sizes, context, context_lens,
                dtype, num_views, cross_view_flex_attn, crossview_attn_type,
                cond_mask, kv_cache_dict, return_kv, skip_cross_attn,
                score_vis_dir, offload_kv_cache)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # head
        x = self.head(x, e, grid_sizes)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()

        if return_kv:
            return x, kv_cache_dict_ret
        else:
            return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs={},
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        logger.info(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate

                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    logger.info(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        logger.info(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )

                return model
            except Exception as e:
                logger.warning(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight']
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']

        tmp_state_dict = {}
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                logger.warning(f"{key} Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        logger.warning(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        model = model.to(torch_dtype)
        return model
