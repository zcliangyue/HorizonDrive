# shared base settings for eval

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=".:$PYTHONPATH"
export DEFAULT_OUTPUT_DIR="./logs/"


RUNTIME_ARGS=(
    --train_mode="unified"
    --mixed_precision="bf16"
    --seed=42
    --crossview_attn_type="full"
)

export DEFAULT_ARGS=(
    ${RUNTIME_ARGS[@]}
    --low_vram
)
