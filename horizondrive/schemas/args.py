import argparse
import os


class Args:

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser(description="Eval-only entry for horizondrive.")
        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default=None,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="sd-model-finetuned",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible evaluation.")
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Kept for Accelerator initialization compatibility.",
        )
        parser.add_argument(
            "--logging_dir",
            type=str,
            default="logs",
            help="Logging directory under output_dir.",
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help="Mixed precision override for Accelerator.",
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="tensorboard",
            help="Tracker backend passed to Accelerator.",
        )
        parser.add_argument("--local_rank", type=int, default=-1, help="Local rank injected by accelerate.")
        parser.add_argument(
            "--validation_steps",
            type=int,
            default=2000,
            help="Kept for non-final validation scheduling compatibility.",
        )
        parser.add_argument(
            "--tracker_project_name",
            type=str,
            default="text2image-action",
            help="Project name passed to Accelerator.init_trackers.",
        )
        parser.add_argument(
            "--crossview_attn_type",
            type=str,
            default="full",
            choices=["full", "loop", "flex", "blockwise_causal", "window_causal"],
        )
        parser.add_argument(
            "--config_path",
            type=str,
            default=None,
            help=(
                "The config of the model in training."
            ),
        )
        parser.add_argument(
            "--transformer_path",
            type=str,
            default=None,
            help=("If you want to load the weight from other transformers, input its path."),
        )
        parser.add_argument(
            "--vae_path",
            type=str,
            default=None,
            help=("If you want to load the weight from other vaes, input its path."),
        )
        parser.add_argument(
            "--low_vram", action="store_true", help="Whether enable low_vram mode."
        )
        parser.add_argument(
            "--train_mode",
            type=str,
            default="normal",
            help=(
                'The format of training data. Support `"normal"`'
                ' (default), `"i2v"`.'
            ),
        )
        parser.add_argument(
            "--debugpy",
            action="store_true",
            help="Whether to use debugpy.",
        )

        args = parser.parse_args()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank
        return args
