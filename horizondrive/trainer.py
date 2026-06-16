"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
import logging
import os
import sys
import json
import math
from typing import Union

import accelerate
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from typing_extensions import override
from einops import rearrange
import datasets
from copy import deepcopy

from videox_fun.utils.config import Config

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from torch.utils.data import DataLoader, Dataset
from horizondrive.schemas.components import Components
from horizondrive.schemas.state import State
from horizondrive.utils.memory_utils import get_memory_statistics, free_memory
from horizondrive.utils.metric_utils import (
    StyleGanFVDMetric,
    FIDDistMetric,
)
from horizondrive.utils.train_utils import WAN_FUN_NEGATIVE_PROMPT
from videox_fun.models import CLIPModel, WanT5EncoderModel, UnifiedTransformer3DModel
from videox_fun.utils.utils import (
    construct_emb_cls,
    filter_kwargs,
    import_cls,
    save_multiview_videos_grid,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

LOG_NAME = "trainer"
LOG_LEVEL = "INFO"
logger = get_logger(LOG_NAME, LOG_LEVEL)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.state: State = State()
        self.config = Config.fromfile(self.args.config_path).to_omegaconf()
        # load args from config
        if self.config.get("args", None) is not None:
            for key, value in self.config.args.items():
                if not hasattr(self.args, key):
                    setattr(self.args, key, value)
        
        # update args with config trainer_kwargs
        if self.config.get("trainer_kwargs", None) is not None:
            for key, value in self.config.trainer_kwargs.items():
                setattr(self.args, key, value)
        
        self.components = Components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self.components = self.load_components()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _normalize_validation_prompt(self, prompt, num_views: int):
        if isinstance(prompt, dict):
            return [prompt[k] for k in sorted(prompt.keys())]
        if isinstance(prompt, list):
            return prompt
        return prompt

    def _expand_validation_prompt_embeds(self, prompt_embeds: torch.Tensor, num_views: int) -> torch.Tensor:
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.shape[0] == 1 and num_views > 1:
            repeat_dims = [num_views] + [1] * (prompt_embeds.ndim - 1)
            prompt_embeds = prompt_embeds.repeat(*repeat_dims)
        return prompt_embeds

    def _crop_condition_frames_for_metrics(
        self,
        videos: torch.Tensor,
        num_views: int,
        num_condition_images: int,
    ) -> torch.Tensor:
        if num_condition_images <= 0:
            return videos
        if videos.shape[0] % num_views != 0:
            raise ValueError(
                f"Cannot crop metric frames: video frame count {videos.shape[0]} "
                f"is not divisible by num_views {num_views}."
            )
        frames_per_view = videos.shape[0] // num_views
        if num_condition_images >= frames_per_view:
            raise ValueError(
                f"Cannot crop {num_condition_images} condition frames from "
                f"{frames_per_view} frames per view."
            )
        videos = rearrange(videos, "(v t) c h w -> v t c h w", v=num_views)
        videos = videos[:, num_condition_images:].contiguous()
        return rearrange(videos, "v t c h w -> (v t) c h w").contiguous()

    def _init_distributed(self):
        args = self.args
        # logging_dir = "/job_tboard"
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        # print('logging_dir', logging_dir)
        os.makedirs(logging_dir, exist_ok=True)
        save_config_path = os.path.join(logging_dir, "model_param.yaml")
        with open(save_config_path, 'w') as f:
            OmegaConf.save(self.config, f)

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        # deepspeed_plugin = None
        if deepspeed_plugin is not None:
            self.state.zero_stage = int(deepspeed_plugin.zero_stage)
            print(f"Using DeepSpeed Zero stage: {self.state.zero_stage}")
        else:
            self.state.zero_stage = 0
            print("DeepSpeed is not enabled.")

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)
            self.state.rng = np.random.default_rng(np.random.PCG64(args.seed + self.accelerator.process_index))
            self.state.torch_rng = torch.Generator(self.accelerator.device).manual_seed(args.seed + self.accelerator.process_index)
        else:
            self.state.rng = None
            self.state.torch_rng = None
        self.state.index_rng = np.random.default_rng(np.random.PCG64(43))
        print(f"Init rng with seed {args.seed + self.accelerator.process_index}. Process_index is {self.accelerator.process_index}")

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.state.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.state.weight_dtype = torch.float16
            self.args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.state.weight_dtype = torch.bfloat16
            self.args.mixed_precision = self.accelerator.mixed_precision


    def _init_logging(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()


    def _init_directories(self):
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

    def prepare_trackers(self):
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            tracker_config.pop("val_data_meta", None)
            tracker_config.pop("trainable_modules", None)
            tracker_config.pop("trainable_modules_low_learning_rate", None)
            
            # Keep only tracker config values supported by the logger backend.
            tracker_config_tmp = deepcopy(tracker_config)
            for key, value in tracker_config.items():
                if not isinstance(value, (int, float, str, bool, torch.Tensor)):
                    tracker_config_tmp.pop(key)
            
            self.accelerator.init_trackers(self.args.tracker_project_name, tracker_config_tmp)
    
    def eval(self):
        self.prepare_for_validation(clip_validation_set=False)
        self.prepare_trackers()
        self.validate("final")

    def prepare_for_validation(self, clip_validation_set=True):
        raise NotImplementedError

    def validate(self, step: Union[int, str]) -> None:
        raise NotImplementedError

    def load_components(self):
        raise NotImplementedError

    def initialize_pipeline(self):
        raise NotImplementedError


class WanUnifiedTrainer(Trainer):
    @override
    def load_components(self):
        components = Components()
        config = self.config
        args = self.args

        components.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        components.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        components.transformer3d = UnifiedTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(self.state.weight_dtype)

        if "T2V" not in args.pretrained_model_name_or_path:
            components.clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            components.clip_image_encoder = components.clip_image_encoder.eval()

        if args.transformer_path is not None:
            print(f"From checkpoint: {args.transformer_path}")

            if args.transformer_path.endswith("safetensors"):
                # Support both Wan safetensors checkpoints and legacy .pt exports.
                from safetensors.torch import load_file
                state_dict = load_file(args.transformer_path)
            else:
                ckpt = torch.load(args.transformer_path, map_location="cpu")

                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "module" in ckpt:
                    state_dict = ckpt["module"]
                else:
                    state_dict = ckpt

            model_state_dict = components.transformer3d.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for key, value in state_dict.items():
                load_key = key
                for prefix in ("module.", "transformer3d.", "model.", "student."):
                    if load_key.startswith(prefix):
                        load_key = load_key[len(prefix):]
                # Older checkpoints may still contain branches removed from the
                # open-source eval model. Load only keys that still exist and match shape.
                if load_key in model_state_dict and model_state_dict[load_key].size() == value.size():
                    filtered_state_dict[load_key] = value
                else:
                    skipped_keys.append(key)

            if len(skipped_keys) > 0:
                print(f"skip {len(skipped_keys)} checkpoint keys not used by current model")
                skipped_prefixes = sorted({key.split(".")[0] for key in skipped_keys})
                print(f"skipped key prefixes: {skipped_prefixes}")
                print("first skipped keys:")
                for key in skipped_keys[:50]:
                    print(f"  {key}")

            m, u = components.transformer3d.load_state_dict(filtered_state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        def deepspeed_zero_init_disabled_context_manager():
            """
            Return a context manager list that disables zero.Init when needed.
            """
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            components.text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=self.state.weight_dtype,
            )
            components.text_encoder = components.text_encoder.eval()
            if args.vae_path is not None:
                vae_path = args.vae_path
            else:
                vae_path = os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'horizondrive-vae.pkl'))
            AutoencoderKL = import_cls(config['vae_kwargs'].get('vae_type', 'videox_fun.models.AutoencoderKLWan'))
            components.vae = AutoencoderKL.from_pretrained(
                vae_path,
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            )

        components.print_components()

        return components

    @override
    def initialize_pipeline(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(self.config['scheduler_kwargs']))
        )
        if self.args.low_vram and self.args.train_mode != "normal" and self.components.clip_image_encoder is not None:
            self.components.clip_image_encoder.to(self.accelerator.device)
        pipeline = import_cls(self.config['pipeline'])(
            vae=self.accelerator.unwrap_model(self.components.vae).to(self.state.weight_dtype),
            text_encoder=self.accelerator.unwrap_model(self.components.text_encoder),
            tokenizer=self.components.tokenizer,
            transformer=self.accelerator.unwrap_model(self.components.transformer3d),
            scheduler=scheduler,
            clip_image_encoder=self.components.clip_image_encoder,
        )

        return pipeline

    def prepare_for_validation(self, clip_validation_set=True):
        """
        Prepare validation data by loading metadata file and extracting first frames.
        """
        validation_kwargs = self.config.get("validation_kwargs", {})
        dataset_kwargs = validation_kwargs.get("dataset_kwargs", {})
        val_datasets = {}
        for dataset_name, dataset_config in dataset_kwargs.items():
            val_datasets[dataset_name] = construct_emb_cls(dataset_config)
        self.val_datasets = val_datasets

        self.val_modes = validation_kwargs.get("val_modes", {})
        self.eval_metrics = validation_kwargs.get("eval_metrics", ["fid"])
        if "max_validation_samples" in validation_kwargs:
            self.max_validation_samples = validation_kwargs["max_validation_samples"]
        else:
            max_samples_per_gpu = validation_kwargs.get("max_samples_per_gpu", 1)
            self.max_validation_samples = max_samples_per_gpu * self.accelerator.num_processes

    @torch.no_grad()
    def validate(self, step: Union[int, str]) -> None:
        if len(self.val_modes) == 0:
            logger.warning("No validation modes configured. Skipping validation.")
            return

        if step != "final" and len(self.val_modes) > 1:
            sample_names = []
            for name, cfg in self.val_modes.items():
                every_n = cfg.get("every_n", 1)
                if step % (self.args.validation_steps * every_n) == 0:
                    sample_names.append(name)
        else:
            sample_names = list(self.val_modes.keys())

        logger.info(f"Validating {len(sample_names)} modes: {sample_names}...")

        self.components.transformer3d.eval()
        torch.set_grad_enabled(False)

        pipeline = self.initialize_pipeline()
        pipeline = pipeline.to(self.accelerator.device)
        logger.info(f"Process {self.accelerator.process_index} Pipeline initialized")

        if hasattr(pipeline.transformer, "_compiled_with_torch_compile"):
            logger.info("torch.compile warmup: running dummy forward pass...")
            device = self.accelerator.device
            dtype = self.state.weight_dtype
            num_frames, height, width = 21, 384, 768
            t_cr = self.components.vae.config.temporal_compression_ratio
            s_cr = self.components.vae.config.spatial_compression_ratio
            num_cameras = 1
            latent_frames = ((num_frames - 1) // t_cr + 1) * num_cameras
            latent_h = height // s_cr
            latent_w = width // s_cr
            patch_size = pipeline.transformer.config.patch_size
            seq_len = math.ceil(latent_h * latent_w / (patch_size[1] * patch_size[2]) * latent_frames)
            dummy_x = torch.randn(1, pipeline.transformer.config.in_channels, latent_frames, latent_h, latent_w, device=device, dtype=dtype)
            dummy_t = torch.tensor([0.5], device=device, dtype=dtype)
            dummy_context = [torch.randn(512, pipeline.transformer.config.text_dim, device=device, dtype=dtype)]
            pipeline.transformer(
                x=dummy_x, context=dummy_context, t=dummy_t, seq_len=seq_len,
                num_views=num_cameras, dtype=dtype,
            )
            del dummy_x, dummy_t, dummy_context
            torch.cuda.empty_cache()
            logger.info("torch.compile warmup done.")

        for i, sample_name in enumerate(sample_names):
            dataset_name = self.val_modes[sample_name]["dataset_name"]
            dataset_kwargs = dict(self.val_modes[sample_name].get("kwargs", {}))
            logger.info(f"Validating {i+1}/{len(sample_names)} mode: {sample_name}... dataset_name: {dataset_name}, dataset_kwargs: {dataset_kwargs}")
            self.validate_one_mode(
                pipeline=pipeline,
                validation_mode=sample_name,
                val_dataset=self.val_datasets[dataset_name],
                val_dataset_kwargs=dataset_kwargs,
                step=step,
                max_samples=self.max_validation_samples,
                pipeline_kwargs=dict(self.val_modes[sample_name].get("pipeline_kwargs", {})),
            )

        del pipeline
        if self.args.low_vram:
            if self.args.train_mode != "normal" and self.components.clip_image_encoder is not None:
                self.components.clip_image_encoder.to('cpu')
            self.components.vae.to('cpu')
            self.components.text_encoder.to('cpu')

        torch.set_grad_enabled(True)
        self.components.transformer3d.train()

        free_memory()

    @torch.no_grad()
    def validate_one_mode(
        self,
        pipeline,
        validation_mode: str,
        val_dataset,
        val_dataset_kwargs,
        pipeline_kwargs,
        step: Union[int, str],
        max_samples: int,
    ) -> None:
        """
        Perform validation using pre-extracted images with distributed processing.
        """
        accelerator = self.accelerator
        weight_dtype = self.state.weight_dtype
        logger.info(f"Process {accelerator.process_index} Starting validation...")
        free_memory()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        if self.args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(self.args.seed)

        validation_path = os.path.join(self.args.output_dir, f"validation_res_{step}", validation_mode)
        if accelerator.is_main_process:
            os.makedirs(validation_path, exist_ok=True)
        accelerator.wait_for_everyone()

        total_samples = min(len(val_dataset), max_samples)
        if hasattr(val_dataset, "temporal_compression_ratio"):
            val_dataset.temporal_compression_ratio = self.components.vae.config.temporal_compression_ratio

        samples = torch.arange(total_samples)
        local_samples = samples[accelerator.process_index::accelerator.num_processes]

        logger.info(
            f"Process {accelerator.process_index} processing {len(local_samples)} validation samples "
            f"(global total: {total_samples}, num_processes: {accelerator.num_processes})",
            main_process_only=False
        )
        if "fid" in self.eval_metrics:
            fid_metric_cls = FIDDistMetric(normalize=True, device=accelerator.device)
        if "fvd" in self.eval_metrics:
            fvd_metric_cls = StyleGanFVDMetric(device=accelerator.device)
            fvd_segment_count = 0

        processed_count = 0
        num_views = len(val_dataset.camera_names)
        for idx_tensor in local_samples:
            idx = idx_tensor.item()
            val_dataset_kwargs["idx"] = idx
            val_dataset_kwargs["validation_mode"] = validation_mode
            if validation_mode.startswith("i2v") and "num_condition_images" in pipeline_kwargs:
                val_dataset_kwargs["num_condition_images"] = pipeline_kwargs["num_condition_images"]

            pipe_pipeline_kwargs = dict(pipeline_kwargs)
            infer_image_len = pipe_pipeline_kwargs.pop("infer_image_len", None)
            if infer_image_len is not None:
                if infer_image_len == -1:
                    nf = val_dataset_kwargs["resolution"][0]
                    n_cond = pipe_pipeline_kwargs.get("num_condition_images", 1)
                    n_unroll = pipe_pipeline_kwargs.get("num_unroll_steps", 1)
                    infer_image_len = nf + max(n_unroll - 1, 0) * (nf - n_cond)
                val_dataset_kwargs["infer_image_len"] = infer_image_len

            sample_data = val_dataset.__getitem__(val_dataset_kwargs)
            if sample_data is None:
                continue
            prompt = self._normalize_validation_prompt(sample_data['text'], num_views)
            logger.info(
                f"Process {accelerator.process_index} validating sample {idx+1}/{total_samples}. "
                f"Prompt: {prompt}",
                main_process_only=False,
            )
            model_mode = sample_data["model_mode"]
            clip_id = f"{idx}_{sample_data['clip_id']}"
            metric_info = {}

            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    num_frames, height, width = val_dataset_kwargs["resolution"]
                    video_length = int((num_frames - 1) // self.components.vae.config.temporal_compression_ratio * self.components.vae.config.temporal_compression_ratio) + 1 if num_frames != 1 else 1
                    for key, value in sample_data['conditions'].items():
                        sample_data['conditions'][key] = value.unsqueeze(0)

                    vis_additional_conditions = {}
                    for con_name, con_value in sample_data['conditions'].items():
                        if con_name in ["bbox", "hdmap"]:
                            vis_additional_conditions[con_name] = (con_value[0].clone() + 1.0) / 2.0

                    mask_video, mask, clip_image = None, None, None
                    if model_mode == "i2v":
                        mask = sample_data['mask'].unsqueeze(0)
                        mask_video = sample_data.get('mask_pixel_values')
                        if mask_video is not None:
                            mask_video = mask_video.unsqueeze(0)
                        clip_image = sample_data.get('clip_pixel_values')

                    prompt_embeds = None
                    negative_prompt_embeds = None
                    cached_prompt_embeds = sample_data.get("encoder_hidden_states")
                    if cached_prompt_embeds is not None:
                        prompt_embeds = self._expand_validation_prompt_embeds(
                            cached_prompt_embeds.to(accelerator.device, dtype=weight_dtype),
                            num_views,
                        )
                        prompt = None
                        cached_negative_prompt_embeds = sample_data.get("negative_encoder_hidden_states")
                        if cached_negative_prompt_embeds is not None:
                            negative_prompt_embeds = self._expand_validation_prompt_embeds(
                                cached_negative_prompt_embeds.to(accelerator.device, dtype=weight_dtype),
                                num_views,
                            )

                    negative_prompt = None if negative_prompt_embeds is not None else WAN_FUN_NEGATIVE_PROMPT

                    pipe_kwargs = {
                        "prompt": prompt,
                        "prompt_embeds": prompt_embeds,
                        "video": sample_data["pixel_values"].unsqueeze(0),
                        "mask_video": mask_video,
                        "mask": mask,
                        "clip_image": clip_image,
                        "num_frames": video_length,
                        "negative_prompt": negative_prompt,
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "height": height,
                        "width": width,
                        "guidance_scale": self.config.get("validation_kwargs", {}).get("guidance_scale", 1.0),
                        "generator": generator,
                        "num_cameras": num_views,
                        "additional_conditions": sample_data['conditions'],
                        "crossview_attn_type": self.args.crossview_attn_type,
                        "step": step,
                        "t_compression_ratio": self.components.vae.config.temporal_compression_ratio,
                        "use_t_variant_noise": self.config.get("trainer_kwargs", {}).get("use_t_variant_noise", True),
                        "num_inference_steps": len(self.args.denoising_step_indices_list),
                        "sigmas": np.asarray(self.args.denoising_step_indices_list, dtype=np.float32)
                        / self.config.get("scheduler_kwargs", {}).get("num_train_timesteps", 1000),
                        **pipe_pipeline_kwargs,
                        **self.config.get("extra_pipeline_kwargs", {}),
                    }

                    videos = pipeline(**pipe_kwargs).videos
                    videos = rearrange(videos[0], "c f h w -> f c h w").contiguous()

                gt_videos = (sample_data['pixel_values'] + 1.0) / 2.0
                metric_num_condition_images = pipe_pipeline_kwargs.get("num_condition_images", 0)
                metric_videos = self._crop_condition_frames_for_metrics(
                    videos,
                    num_views=num_views,
                    num_condition_images=metric_num_condition_images,
                )
                metric_gt_videos = self._crop_condition_frames_for_metrics(
                    gt_videos,
                    num_views=num_views,
                    num_condition_images=metric_num_condition_images,
                )
                for eval_metric in self.eval_metrics:
                    if eval_metric == "fid" or eval_metric == "fvd":
                        if eval_metric == "fid":
                            fid_metric_cls.update(metric_videos, real=False)
                            fid_metric_cls.update(metric_gt_videos, real=True)
                        if eval_metric == "fvd":
                            fvd_segment_count += fvd_metric_cls.update(
                                metric_videos,
                                real=False,
                                video_length=16,
                                segment_stride=16,
                                num_views=num_views,
                            )
                            fvd_metric_cls.update(
                                metric_gt_videos,
                                real=True,
                                video_length=16,
                                segment_stride=16,
                                num_views=num_views,
                            )

                if len(metric_info) > 0:
                    metric_file = os.path.join(validation_path, f"{clip_id}_metrics.json")
                    with open(metric_file, 'w') as f:
                        json.dump(metric_info, f, indent=4)

                filename = os.path.join(validation_path, f"{clip_id}.mp4")
                gt_filename = os.path.join(validation_path, f"{clip_id}_gt.mp4")
                comparison_filename = os.path.join(validation_path, f"{clip_id}_comparison.mp4")

                save_multiview_videos_grid(
                    videos,
                    comparison_filename,
                    fps=10,
                    gt_videos=gt_videos,
                    camera_names=val_dataset.camera_names,
                    fid_metric=metric_info["fid_score"] if "fid_score" in metric_info else None,
                )

                save_multiview_videos_grid(
                    videos,
                    filename,
                    fps=10,
                    camera_names=val_dataset.camera_names,
                    fid_metric=metric_info["fid_score"] if "fid_score" in metric_info else None,
                )
                save_multiview_videos_grid(gt_videos, gt_filename, fps=10, camera_names=val_dataset.camera_names)

                # Save condition visualizations next to the generated sample.
                if len(vis_additional_conditions) > 0:
                    for con_name, con_value in vis_additional_conditions.items():
                        save_multiview_videos_grid(
                            con_value,
                            os.path.join(validation_path, f"{clip_id}_{con_name}.mp4"),
                            gt_videos=videos,
                            fps=10,
                            camera_names=val_dataset.camera_names,
                            overlap=True,
                        )

            processed_count += 1

        if "fid" in self.eval_metrics and total_samples > 0:
            fid = fid_metric_cls.compute()
            del fid_metric_cls
            if accelerator.is_main_process:
                all_fid_path = os.path.join(validation_path, "fid_scores.txt")
                with open(all_fid_path, 'w') as f:
                    f.write(f"All FID scores: {fid.item()}\n")

        if "fvd" in self.eval_metrics and total_samples > 0:
            fvd = fvd_metric_cls.compute()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered_counts = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(gathered_counts, fvd_segment_count)
                all_fvd_segment_count = sum(gathered_counts)
            else:
                all_fvd_segment_count = fvd_segment_count
            if accelerator.is_main_process:
                all_fvd_path = os.path.join(validation_path, "fvd_scores.txt")
                with open(all_fvd_path, 'w') as f:
                    f.write(f"All FVD scores: {fvd.item()}\n")
                    f.write(f"Num 16-frame segments: {all_fvd_segment_count}\n")
            del fvd_metric_cls

        accelerator.wait_for_everyone()

        logger.info(
            f"Process {accelerator.process_index} completed validation. Processed {processed_count} samples.",
            main_process_only=False
        )

        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        if accelerator.is_main_process:
            logger.info(f"Validation completed successfully. Results saved to {validation_path}")
