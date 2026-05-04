import argparse
import os

def parse_args(sys_args=None):
    """Parse command line arguments for SatMAE finetune"""
    
    parser = argparse.ArgumentParser(description="SatMAE Finetune")
    
    # Dataset arguments
    parser.add_argument("--temporal_config", type=str, required=True, 
                        help="Temporal type of the dataset, S: static, SH: short horizon temporal, LH: long horizon temporal, CD: change detection", 
                        choices=["S", "SH", "LH", "CD"])
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_version", type=str, default=None, help="Version of the dataset")
    parser.add_argument("--task_type", type=str, choices=["classification", "multilabel", "segmentation", "regression"], required=True, help="Task type")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to the GFMBench")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses to use for data loading")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Whether to pin memory for data loading")
    parser.add_argument("--use_8bit", action="store_true", help="Whether to use 8-bit data loading")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop size for training")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--scale", type=float, default=None, help="Scale for training")
    parser.add_argument("--random_crop", action="store_true", help="Whether to use random crop for training")
    parser.add_argument("--random_rotation", action="store_true", help="Whether to use random rotation for training")
    parser.add_argument("--random_resize", action="store_true", help="Whether to use random resize for training")
    parser.add_argument("--train_frac", type=float, default=1.0, help="Fraction of train set to be used in training")
    parser.add_argument("--val_frac", type=float, default=1.0, help="Fraction of val set to be used in evaluation")
    parser.add_argument("--test_frac", type=float, default=1.0, help="Fraction of test set to be used in testing")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--zero_mean", action="store_true", help="Whether to use zero mean for training")
    
    # Temporal arguments
    parser.add_argument("--frames_fliter", type=int, default=1, help="Frames to filter")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of frames to use")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=4, help="Size of patches for hyperspectral patch embedding")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--decoder_model", type=str, default=None, help="Decoder model for segmentation (convnet or upernet)")
    
    # Temporal model arguments
    parser.add_argument("--temporal_model", action="store_true", help="Use temporal model")
    parser.add_argument("--temporal_embedding", action="store_true", help="Use temporal embedding")
    parser.add_argument("--temporal_pooling", type=str, default="mean", 
                       choices=["mean", "max", "attention", "diff", "pretrain"], 
                       help="Temporal pooling method")
    
    # Training arguments
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--early_stop_steps", type=int, default=None, help="Early stop steps")
    
    # Evaluation arguments
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--eval_strategy", type=str, choices=["epoch", "steps", "no"], 
                       default="epoch", help="Evaluation strategy")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints and logs")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--report_to", type=str, default="wandb", help="Where to report results to (tensorboard, wandb, etc.)")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="If set, deletes the older checkpoints in output_dir")
    parser.add_argument("--wandb_dir", type=str, default="wandb", help="Directory to save wandb logs")
    
    # Other arguments
    parser.add_argument("--lp", action="store_true", help="Whether to use linear probe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, choices=["None", "fp16", "bf16"], 
                       default="None", help="Mixed precision")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    # Append run name to directories
    args = parser.parse_args(sys_args)
    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.run_name)
    args.logging_dir = os.path.join(args.logging_dir, args.dataset_name)
    return args
