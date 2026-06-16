#!/bin/bash

source $(dirname $(realpath $0))/_base.sh
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export TORCH_HOME=${TORCH_HOME:-/tmp/torch_cache}

export OUTPUT_DIR=$DEFAULT_OUTPUT_DIR/test/
export MODEL_NAME=${MODEL_NAME:-"models/Wan2.1-T2V-1.3B"}
export EVAL_CKPT=${EVAL_CKPT:-"models/horizondrive-dit.safetensors"}
export VAE_PATH=${VAE_PATH:-"models/horizondrive-vae.pkl"}
export NUM_PROCESSES=${NUM_PROCESSES:-1}
export NUM_MACHINES=${NUM_MACHINES:-1}
export CONFIG_PATH=${CONFIG_PATH:-"config/horizondrive_nuscenes_eval.yaml"}
export ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-"bf16"}
export ACCELERATE_DYNAMO_BACKEND=${ACCELERATE_DYNAMO_BACKEND:-"no"}

if [[ -z "${MAIN_PROCESS_PORT:-}" ]]; then
    MAIN_PROCESS_PORT=$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
fi
export MAIN_PROCESS_PORT

EXTRA_ARGS=(
    --config_path="${CONFIG_PATH}"
    --pretrained_model_name_or_path="${MODEL_NAME}"
    --output_dir="${OUTPUT_DIR}"
    --vae_path="${VAE_PATH}"
    --transformer_path="${EVAL_CKPT}"
)

set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --num_machines="${NUM_MACHINES}" \
    --main_process_port="${MAIN_PROCESS_PORT}" \
    --mixed_precision="${ACCELERATE_MIXED_PRECISION}" \
    --dynamo_backend="${ACCELERATE_DYNAMO_BACKEND}" \
    -- \
    horizondrive/eval.py \
    "${DEFAULT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    $@
