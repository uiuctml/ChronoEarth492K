#!/bin/bash
# Run temporal MAE smoke tests for all (or selected) baselines.
#
# Usage:
#   bash temporal_pretrain/run_tests.sh              # test all 4 models
#   bash temporal_pretrain/run_tests.sh dofa satmae  # test subset

set -e

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/EO1H}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/baseline_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/temporal_test_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false   # suppress HF warning

# Which models to test — default to all, override with positional args
MODELS="${*:-dofa satmae specvit lessvit}"

echo "============================================"
echo "  Temporal MAE Smoke Test"
echo "  Models : $MODELS"
echo "  Data   : $DATA_DIR"
echo "  Output : $OUTPUT_DIR"
echo "============================================"

python -m temporal_pretrain.test_all \
    --data_dir  "$DATA_DIR"   \
    --model_dir "$MODEL_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS
