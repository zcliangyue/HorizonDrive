#!/bin/bash
set -e

export CONFIG_PATH=${CONFIG_PATH:-"config/horizondrive_nuscenes_eval.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export NUM_PROCESSES=${NUM_PROCESSES:-1}

bash horizondrive/shell/eval.sh "$@"
