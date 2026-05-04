import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 Temporal MAE Pretraining")

    # Dataset
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--regions", type=str, nargs="+",
                        default=["AC", "AF", "EA", "EU", "LA", "NA", "OC", "SEA", "SWA"])
    parser.add_argument("--channel_groups", type=str, nargs="+",
                        default=["VNIR", "SWIR1", "SWIR2", "SWIR3", "SWIR4"])
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Max frames per location to sample")
    parser.add_argument("--frames_lb", type=int, default=2,
                        help="Minimum frames a location must have to be included")
    parser.add_argument("--max_frames_budget", type=int, default=None,
                        help="Total frame slots per batch for AdaptiveBucketBatchSampler. "
                             "Defaults to per_device_train_batch_size * num_frames.")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--slow_sample_threshold", type=float, default=10.0,
                        help="Log TemporalChronoEarth samples that take at least this many seconds. "
                             "Set 0 to disable.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--dataloader_persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across epochs.")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2,
                        help="Number of batches prefetched per DataLoader worker.")

    # Stage-1 checkpoint (frozen encoder)
    parser.add_argument("--stage1_model", type=str, required=True,
                        choices=["dofa", "satmae", "specvit", "lessvit"],
                        help="Which stage-1 model to use as the frozen backbone.")
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to stage-1 pretrained weights directory")

    # Temporal MAE architecture
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--n_temporal_layers", type=int, default=4,
                        help="Number of temporal transformer blocks")
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--decoder_depth", type=int, default=2,
                        help="Number of shallow ViT decoder blocks")
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--temporal_mask_ratio", type=float, default=0.5,
                        help="Deprecated: temporal pretraining now reconstructs only the last valid frame")
    parser.add_argument("--norm_pix_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--channel_dropout", type=float, nargs="+", default=None,
                        help="LESSViT HCS/channel dropout ratio. One value uses a fixed "
                             "dropout; two values sample a dropout ratio range per channel group.")

    # Timestamp embedding
    parser.add_argument("--year_min", type=int, default=2000)
    parser.add_argument("--year_max", type=int, default=2020)

    # Training
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_dir", type=str, default="wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    return args
