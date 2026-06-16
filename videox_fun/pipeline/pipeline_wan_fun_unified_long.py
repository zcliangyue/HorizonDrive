import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange

from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, UnifiedTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from horizondrive.utils.train_utils import (
    prepare_clip_context,
    prepare_mask_condition,
    batch_encode_vae,
    prepare_action_condition,
    align_first_segment_latents,
    replace_condition_with_gt_latents,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor
    decode_videos: Optional[torch.Tensor] = None
    # Full packed latents (B, C, N*F_lat, H', W') after multi-window latent_rollout concat; omit per-window decode→concat.
    rollout_full_latents: Optional[torch.Tensor] = None


class WanFunUnifiedPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: UnifiedTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer,
            clip_image_encoder=clip_image_encoder, scheduler=scheduler,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        if prompt_embeds is None and prompt is not None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, t_compression_ratio, num_channels_latents, num_frames, num_cameras,
        height, width, dtype, device, generator, latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            ((num_frames - 1) // t_compression_ratio + 1) * num_cameras,
            height // self.vae.config.spatial_compression_ratio,
            width // self.vae.config.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    @staticmethod
    def _rollout_latent_frames_to_skip(total_cond_pixel_frames: int, t_compression_ratio: int) -> int:
        """Per-camera latent frames to drop when appending a window (overlap with previous), from pixel overlap."""
        if total_cond_pixel_frames <= 0 or t_compression_ratio <= 0:
            return 0
        return (total_cond_pixel_frames + t_compression_ratio - 1) // t_compression_ratio

    def decode_latents(self, latents: torch.Tensor, num_cameras=1) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        latents = rearrange(latents, "b c (n f) h w -> (b n) c f h w", n=num_cameras)
        frames = self.vae.decode(latents).sample
        frames = rearrange(frames, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self, prompt, height, width, negative_prompt,
        callback_on_step_end_tensor_inputs, prompt_embeds=None, negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # if prompt_embeds is not None and negative_prompt_embeds is not None:
        #     if prompt_embeds.shape != negative_prompt_embeds.shape:
        #         raise ValueError(
        #             "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
        #             f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
        #             f" {negative_prompt_embeds.shape}."
        #         )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------ #
    #  Single-window denoising
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_one(
        self,
        latents,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        clip_context,
        num_cameras=1,
        cond_latents=None,
        fill_mask=None,
        additional_conditions={},
        crossview_attn_type="full",
        guidance_scale: float = 6,
        use_t_variant_noise=False,
        extra_step_kwargs: Dict[str, Any] = {},
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_type="numpy",
        step=0,
        use_tqdm=True,
        return_latents=False,
        skip_final_decode: bool = False,
    ):
        device = self._execution_device
        weight_dtype = self.text_encoder.dtype
        dtype = latents.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # Fill condition latents into noise
        kv_cache_dict = {}
        if fill_mask is not None:
            if cond_latents is not None:
                latents[fill_mask] = cond_latents.flatten()
            cond_mask = fill_mask[:, 0, :, 0, 0]  # (b, f)
        else:
            cond_mask = None

        seq_len = math.ceil(
            (latents.shape[3] * latents.shape[4])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * latents.shape[2]
        )

        if use_tqdm:
            timesteps_iter = tqdm(timesteps, desc="Denoising")
        else:
            timesteps_iter = timesteps
        for i, t in enumerate(timesteps_iter):
            self.transformer.current_steps = i

            if self.interrupt:
                continue

            if hasattr(self.scheduler, "scale_model_input"):
                latents = self.scheduler.scale_model_input(latents, t)

            timestep = t.expand(latents.shape[0])
            if use_t_variant_noise:
                timestep = timestep.unsqueeze(-1).repeat(1, latents.shape[2]).contiguous()
                if cond_mask is not None:
                    timestep[cond_mask] = 0

            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                if do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        x=latents, context=negative_prompt_embeds, t=timestep,
                        seq_len=seq_len, clip_fea=clip_context, num_views=num_cameras,
                        dtype=dtype, additional_conditions=additional_conditions,
                        crossview_attn_type=crossview_attn_type, step=step,
                        kv_cache_dict=kv_cache_dict, cond_mask=cond_mask,
                    )
                    noise_pred_cond = self.transformer(
                        x=latents, context=prompt_embeds, t=timestep,
                        seq_len=seq_len, clip_fea=clip_context, num_views=num_cameras,
                        dtype=dtype, additional_conditions=additional_conditions,
                        crossview_attn_type=crossview_attn_type, step=step,
                        kv_cache_dict=kv_cache_dict, cond_mask=cond_mask,
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.transformer(
                        x=latents, context=prompt_embeds, t=timestep,
                        seq_len=seq_len, clip_fea=clip_context, num_views=num_cameras,
                        dtype=dtype, additional_conditions=additional_conditions,
                        crossview_attn_type=crossview_attn_type, step=step,
                    )

            self.scheduler._step_index = None
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            latents = latents.to(dtype)

            if use_t_variant_noise and fill_mask is not None:
                latents[fill_mask] = cond_latents.flatten()

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        torch.cuda.empty_cache()
        clean_latents = latents
        if skip_final_decode:
            if not return_latents:
                raise ValueError("skip_final_decode=True requires return_latents=True")
            return None, clean_latents

        if output_type == "numpy":
            video = self.decode_latents(latents, num_cameras=num_cameras)
        elif output_type != "latent":
            video = self.decode_latents(latents, num_cameras=num_cameras)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        if return_latents:
            return video, clean_latents
        return video

    # ------------------------------------------------------------------ #
    #  Main entry — multi-step rollout 
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        mask: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,  # unused, kept for API compatibility
        video: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_cameras: int = 1,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Union[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        shift: int = 5,
        additional_conditions: Dict = {},
        crossview_attn_type: str = "full",
        step=None,
        use_t_variant_noise: bool = False,
        t_compression_ratio: int = 4,
        num_unroll_steps: int = 1,
        num_condition_images: int = 1,
        return_vae_decode_video: bool = False,
        use_gt_condition: bool = True,
        latent_rollout: bool = False,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            video: (b, nf, c, h, w) tensor of input video.
            num_condition_images: Number of rolling condition frames taken from the tail of the
                previous step's output (overlap region).
        Examples:
        Returns:
        """
        assert use_t_variant_noise, "Currently only support use_t_variant_noise = True"
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        total_cond_frames = num_condition_images

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, negative_prompt,
            callback_on_step_end_tensor_inputs, prompt_embeds, negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) // num_cameras
        elif prompt is None:
            batch_size = 1
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length, device=device,
        )

        # Prepare clip context
        if clip_image is not None and self.clip_image_encoder is not None:
            clip_context = prepare_clip_context(self.clip_image_encoder, clip_image, device=device, dtype=dtype)
        else:
            clip_context = None

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Denoising loop
        self.transformer.num_inference_steps = num_inference_steps

        if video is not None and return_vae_decode_video:
            gt_video_latents = batch_encode_vae(video.to(dtype=dtype, device=device), self.vae, num_cameras=num_cameras)
            gt_decode_video = self.decode_latents(gt_video_latents, num_cameras=num_cameras)
            if output_type == "numpy":
                gt_decode_video = gt_decode_video
            elif output_type != "latent":
                gt_decode_video = self.video_processor.postprocess_video(video=gt_decode_video, output_type=output_type)
            if not return_dict:
                gt_decode_video = torch.from_numpy(gt_decode_video)
        else:
            gt_decode_video = None

        # Compute how many rollout steps the GT video supports
        if video is not None and use_gt_condition:
            total_frames = video.shape[1] // num_cameras
            gt_num_unroll_steps = (total_frames - total_cond_frames) // (num_frames - total_cond_frames)
            num_unroll_steps = min(num_unroll_steps, gt_num_unroll_steps)
            print(f"total_frames: {total_frames}, num_unroll_steps: {num_unroll_steps}")
        noise_frame_num = num_frames - total_cond_frames
        logger.info(
            f"[Rollout Config] num_frames={num_frames}, num_cameras={num_cameras}, "
            f"num_condition_images={num_condition_images}, "
            f"total_cond_frames={total_cond_frames}, noise_frame_num={noise_frame_num}, "
            f"num_unroll_steps={num_unroll_steps}"
        )


        need_latents = latent_rollout and num_unroll_steps > 1
        prev_window_latents = None
        gt_video_latents = None  # directly-encoded GT latents for final decode

        final_video = None
        final_latents: Optional[torch.Tensor] = None
        latent_skip_f = 0
        if need_latents:
            latent_skip_f = self._rollout_latent_frames_to_skip(
                total_cond_frames, t_compression_ratio,
            )
            logger.info(
                f"[Rollout-latent] concat-then-decode: per-camera latent frames to skip per window "
                f"(overlap) = {latent_skip_f} (from total_cond_frames={total_cond_frames}, t_comp={t_compression_ratio})"
            )

        for unroll_step in tqdm(range(num_unroll_steps), desc="Rollout"):
            begin_idx = unroll_step * noise_frame_num

            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                t_compression_ratio,
                self.vae.config.latent_channels,
                num_frames, num_cameras,
                height, width, dtype, device, generator,
            )

            # ---- Step 0: encode GT video, build mask ----
            if unroll_step == 0:
                if mask is not None:
                    mask = rearrange(mask, "b (n f) c h w -> (b n) f c h w", n=num_cameras)
                    mask = rearrange(mask[:, :num_frames], "(b n) f c h w -> b (n f) c h w", n=num_cameras)
                    mask = mask.to(dtype=dtype, device=device)
                    mask_latents = prepare_mask_condition(
                        mask, latents, num_cameras=num_cameras, temporal_compression_ratio=t_compression_ratio,
                    )
                    fill_mask = (mask_latents[:, :1] > 0.5).repeat(1, latents.shape[1], 1, 1, 1)
                else:
                    fill_mask = None

                if video is not None:
                    video = rearrange(video, "b (n f) c h w -> (b n) f c h w", n=num_cameras)
                    video = rearrange(video[:, :num_frames], "(b n) f c h w -> b (n f) c h w", n=num_cameras)
                    video = video.to(dtype=dtype, device=device)

                    # Always encode the raw video for final output decode
                    # Pad first frame before encode so the first segment's latent
                    # distribution aligns with subsequent segments (which always have
                    # preceding context frames).  After encode, strip the padded frames.
                    # This padded version is only used for condition filling.
                    gt_video_latents, video_latents = align_first_segment_latents(
                        video, self.vae, num_cameras=num_cameras,
                        t_compression_ratio=t_compression_ratio,
                    )
                    if fill_mask is not None:
                        cond_latents = video_latents[fill_mask].to(dtype=dtype)
                    else:
                        cond_latents = None

                else:
                    cond_latents = None

            # ---- Step > 0: build rolling condition latents ----
            else:
                self.scheduler._step_index = None
                if latent_rollout and prev_window_latents is not None:
                    # Latent-space rollout: slice the last num_condition_images
                    # frames from the previous window's clean latents.
                    prev_lat = rearrange(
                        prev_window_latents,
                        "b c (n f) h w -> b n c f h w", n=num_cameras,
                    )
                    cond_parts = prev_lat[:, :, :, -num_condition_images:]
                    cond_latents_tmp = rearrange(
                        cond_parts, "b n c f h w -> b c (n f) h w",
                    ).to(dtype=dtype)
                    logger.info(
                        f"[Rollout-latent] step={unroll_step}, "
                        f"sliced {num_condition_images} latent frames per camera from tail"
                    )
                    cond_latents = cond_latents_tmp
                else:
                    # Pixel-space fallback: decode previous output → crop overlap → encode
                    video_arr = torch.from_numpy(video).permute(0, 2, 1, 3, 4) * 2.0 - 1.0
                    video_arr = rearrange(video_arr, "b (n f) c h w -> (b n) f c h w", n=num_cameras)
                    logger.info(
                        f"[Rollout-pixel] step={unroll_step}, video_arr per_cam shape={video_arr.shape}, "
                        f"taking last {num_condition_images} frames as cond"
                    )
                    cond_video = rearrange(
                        video_arr[:, -num_condition_images:],
                        "(b n) f c h w -> b (n f) c h w", n=num_cameras,
                    )
                    cond_latents_tmp = batch_encode_vae(
                        cond_video.to(dtype=dtype, device=device), self.vae, num_cameras=num_cameras,
                    ).to(dtype=dtype)
                    cond_latents = cond_latents_tmp

                            # ---- Build additional conditions for this window ----
            # Slice additional_conditions starting from begin_idx.
            cond_begin_idx = begin_idx
            cur_additional_conditions = {}
            for name, value in additional_conditions.items():
                if name in ["normal", "hdmap", "bbox"]:
                    if value.shape[1] >= (cond_begin_idx + num_frames) * num_cameras:
                        cur_value = rearrange(value, "b (n f) c h w -> (b n) f c h w", n=num_cameras)
                        cur_value = cur_value[:, cond_begin_idx:cond_begin_idx + num_frames].clone()
                        cur_value = rearrange(cur_value, "(b n) f c h w -> b (n f) c h w", n=num_cameras)
                        cur_value = batch_encode_vae(
                            cur_value.to(dtype=dtype, device=device), self.vae, num_cameras=num_cameras,
                        )
                        cur_additional_conditions[name] = cur_value.to(dtype=dtype)

            if "action" in additional_conditions:
                action = additional_conditions["action"]
                if action.shape[1] >= (cond_begin_idx + num_frames) * num_cameras:
                    cur_action = rearrange(action, "b (n f) c -> (b n) f c", n=num_cameras)
                    cur_action = cur_action[:, cond_begin_idx:cond_begin_idx + num_frames].clone()
                    cur_action = rearrange(cur_action, "(b n) f c -> b (n f) c", n=num_cameras)
                    cur_action = cur_action.to(dtype=dtype, device=device)
                    cur_additional_conditions["action"] = prepare_action_condition(
                        cur_action, num_cameras=num_cameras, temporal_compression_ratio=t_compression_ratio,
                    )


            # ---- Denoise one window ----
            gen_kwargs = dict(
                latents=latents,
                cond_latents=cond_latents,
                fill_mask=fill_mask,
                additional_conditions=cur_additional_conditions,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timesteps=timesteps,
                clip_context=clip_context,
                num_cameras=num_cameras,
                crossview_attn_type=crossview_attn_type,
                guidance_scale=guidance_scale,
                use_t_variant_noise=use_t_variant_noise,
                step=step,
                extra_step_kwargs=extra_step_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                output_type=output_type,
                use_tqdm=(num_unroll_steps == 1),
                return_latents=need_latents,
                skip_final_decode=need_latents,
            )
            result = self.generate_one(**gen_kwargs)
            if need_latents:
                _, clean_cur = result
                prev_window_latents = clean_cur
            else:
                video = result

            # ---- Assemble output: latent concat + single decode, or pixel concat ----
            if need_latents:
                if unroll_step == 0:
                    final_latents = clean_cur.clone()
                    logger.info(
                        f"[Rollout-latent] step={unroll_step}, init final_latents shape={tuple(final_latents.shape)}"
                    )
                else:
                    cur_n = rearrange(clean_cur, "b c (n f) h w -> b n c f h w", n=num_cameras)
                    prev_n = rearrange(final_latents, "b c (n f) h w -> b n c f h w", n=num_cameras)
                    f_cur = int(cur_n.shape[3])
                    skip = min(latent_skip_f, f_cur)
                    new_part = cur_n[:, :, :, skip:, :, :]
                    merged_n = torch.cat([prev_n, new_part], dim=3)
                    final_latents = rearrange(merged_n, "b n c f h w -> b c (n f) h w")
                    logger.info(
                        f"[Rollout-latent] step={unroll_step}, skip={skip}, "
                        f"merged_latents shape={tuple(final_latents.shape)}"
                    )
            else:
                if unroll_step == 0:
                    final_video = video
                    logger.info(f"[Rollout] step={unroll_step}, final_video shape={final_video.shape}")
                else:
                    # Per-camera concat: split → append new noise frames → merge (decoded pixels)
                    prev_per_cam = rearrange(final_video, "b c (n f) h w -> (b n) c f h w", n=num_cameras)
                    cur_per_cam = rearrange(video, "b c (n f) h w -> (b n) c f h w", n=num_cameras)
                    new_frames = cur_per_cam[:, :, total_cond_frames:]
                    merged = np.concatenate([prev_per_cam, new_frames], axis=2)
                    final_video = rearrange(merged, "(b n) c f h w -> b c (n f) h w", n=num_cameras)
                    logger.info(
                        f"[Rollout] step={unroll_step}, "
                        f"prev_per_cam={prev_per_cam.shape}, new_frames={new_frames.shape}, "
                        f"final_video={final_video.shape}"
                    )

            torch.cuda.empty_cache()

        self.maybe_free_model_hooks()

        rollout_full_latents_cpu: Optional[torch.Tensor] = None
        if need_latents and final_latents is not None:
            # padding_first_frame_n (default 4) ensures first segment condition
            # frames are replaced with GT latents for latent-rollout alignment.
            if gt_video_latents is not None:
                final_latents = replace_condition_with_gt_latents(
                    final_latents, gt_video_latents, num_cameras, latent_skip_f,
                )
            final_video = self.decode_latents(
                final_latents.to(dtype=self.vae.dtype), num_cameras=num_cameras,
            )
            rollout_full_latents_cpu = final_latents.detach().float().cpu()

        if not return_dict:
            final_video = torch.from_numpy(final_video)

        return WanPipelineOutput(
            videos=final_video,
            decode_videos=gt_decode_video,
            rollout_full_latents=rollout_full_latents_cpu,
        )
