ROOT_DIR="${ROOT_DIR:-$(pwd)}"
echo "Current root directory: $ROOT_DIR"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/data/cache}"

DEBUG=True

if [ "$DEBUG" = True ]; then
    report_to="none"
    regions="AC"
else
    report_to="wandb"
    regions="AC AF EA EU LA NA OC SEA SWA"
fi

DECODER_DEPTH=8
EMBED_DIMS=4
RANK=1

accelerate launch pretrain_scripts/train.py \
    --data_dir "${DATA_DIR:-$ROOT_DIR/data/EO1H}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.75 \
    --warmup_ratio 0.05 \
    --report_to $report_to \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --output_dir ./results/models \
    --logging_dir ./results/logs \
    --wandb_dir ./results/ \
    --run_name LESSVIT_b${EMBED_DIMS}_d${DECODER_DEPTH}_r${RANK} \
    --lr_scheduler_type cosine \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_depth $DECODER_DEPTH \
    --decoder_out_chans 155 \
    --use_perception_field_mask \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --drop_path_rate 0.1 \
    --loss_type mse \
    --modal_mode optical \
    --scale 1 \
    --crop_size 128 \
    --init_values 1.0 \
    --rank $RANK \
    --regions $regions
