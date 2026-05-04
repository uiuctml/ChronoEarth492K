ROOT_DIR="${ROOT_DIR:-$(pwd)}"
echo "Current root directory: $ROOT_DIR"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=3

DEBUG=True

if [ "$DEBUG" = True ]; then
    report_to="none"
else
    report_to="wandb"
fi

DATASET_NAME="CLCD"
MODEL_NAME="satmae"
task_type="segmentation"
pretrained_model_path="${PRETRAINED_MODEL_PATH:-$ROOT_DIR/checkpoints/SatMAE/model.safetensors}"
image_dir="${IMAGE_DIR:-$ROOT_DIR/data/EA}"
label_dir="${LABEL_DIR:-$ROOT_DIR/data/CLCD}"
metadata_path="${METADATA_PATH:-$ROOT_DIR/data/CLCD/CLCD_metadata_static.parquet}"

# 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3

for LR in 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3; do
    accelerate launch --num_processes=1 finetune_scripts/finetune/finetune.py \
        --learning_rate $LR \
        --data_dir "${DATA_DIR:-$ROOT_DIR/data}" \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 10 \
        --learning_rate 1e-4 \
        --weight_decay 0.05 \
        --warmup_ratio 0.05 \
        --report_to $report_to \
        --save_steps 100 \
        --save_total_limit 5 \
        --seed 42 \
        --mixed_precision bf16 \
        --dataloader_num_workers 16 \
        --dataloader_pin_memory \
        --output_dir $ROOT_DIR/results_wyx/models/ \
        --logging_dir $ROOT_DIR/results_wyx/logs \
        --wandb_dir $ROOT_DIR/results_wyx/ \
        --run_name SatMAE_${DATASET_NAME}_lr${LR} \
        --lr_scheduler cosine \
        --decoder_depth 8 \
        --max_grad_norm 1.0 \
        --scale 1 \
        --img_size 128 \
        --crop_size 128 \
        --ignore_index 0 \
        --return_dict \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --task_type $task_type \
        --embed_dim 768 \
        --pretrained_model_path $pretrained_model_path

done
