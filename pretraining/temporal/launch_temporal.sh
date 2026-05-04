#!/bin/bash
# Launch stage-2 temporal MAE pretraining for a given baseline encoder.
#
# Usage:
#   bash temporal_pretrain/launch_temporal.sh dofa
#   bash temporal_pretrain/launch_temporal.sh satmae
#   bash temporal_pretrain/launch_temporal.sh specvit
#   bash temporal_pretrain/launch_temporal.sh lessvit

set -e

MODEL=${1:?"Usage: $0 <dofa|satmae|specvit|lessvit> [gpu_id]"}
GPU=${2:-0}   # default GPU 0; override with e.g. "bash launch_temporal.sh dofa 5"

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/EO1H}"
MODEL_BASE="${MODEL_BASE:-$ROOT_DIR/checkpoints/baseline_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/temporal_pretrain}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/results}"

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU

# ---- Checkpoint path per model ----
case $MODEL in
    dofa)    CKPT="$MODEL_BASE/DOFA" ;;
    satmae)  CKPT="$MODEL_BASE/SatMAE" ;;
    specvit) CKPT="$MODEL_BASE/SpectralViT" ;;
    lessvit) CKPT="$MODEL_BASE/LESSViT/LESSVIT_b2_d8_r1" ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

RUN_NAME="${MODEL}_temporal_stage2"
EXTRA_ARGS=()
if [ "$MODEL" = "lessvit" ]; then
    # HCS from LESSViT static pretraining: randomly keep 25-50% of channels
    # per channel group for the frozen encoder. The reconstruction target stays
    # full-channel in temporal_mae.py.
    EXTRA_ARGS+=(--channel_dropout 0.6 0.7)
fi

echo "============================================"
echo "  Stage-2 temporal pretraining"
echo "  Model   : $MODEL"
echo "  Run     : $RUN_NAME"
echo "  Data    : $DATA_DIR"
echo "  Checkpoint: $CKPT"
echo "============================================"

python -m temporal_pretrain.train \
    --data_dir          "$DATA_DIR"          \
    --stage1_model      "$MODEL"             \
    --stage1_checkpoint "$CKPT"              \
    --regions           AC AF EA EU LA NA OC SEA SWA \
    --channel_groups    VNIR SWIR1 SWIR2 SWIR3 SWIR4 \
    --frames_lb         3                    \
    --num_frames        8                    \
    --crop_size         128                  \
    --embed_dim         768                  \
    --n_temporal_layers 4                    \
    --num_heads         12                   \
    --mlp_ratio         4.0                  \
    --decoder_embed_dim 512                  \
    --decoder_depth     2                    \
    --decoder_num_heads 8                    \
    --decoder_mlp_ratio 4.0                  \
    --norm_pix_loss                          \
    --year_min          2000                 \
    --year_max          2020                 \
    --per_device_train_batch_size 8         \
    --gradient_accumulation_steps 16         \
    --learning_rate     1e-4                 \
    --weight_decay      0.05                 \
    --num_train_epochs  50                  \
    --warmup_ratio      0.1                  \
    --lr_scheduler_type cosine              \
    --max_grad_norm     1.0                  \
    --mixed_precision   bf16                 \
    --dataloader_num_workers 8               \
    --dataloader_pin_memory                  \
    --dataloader_persistent_workers          \
    --dataloader_prefetch_factor 2           \
    --slow_sample_threshold 10               \
    --save_strategy     epoch                \
    --save_total_limit  3                    \
    --report_to         wandb                \
    --wandb_dir         "$WANDB_DIR"         \
    --output_dir        "$OUTPUT_DIR"        \
    --run_name          "$RUN_NAME"          \
    "${EXTRA_ARGS[@]}"
