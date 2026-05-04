import os
import logging
import math
from functools import partial

from accelerate.logging import get_logger
from transformers import is_wandb_available
from transformers import TrainingArguments

from ChronoEarth import get_chronoearth_metadata, NUM_CHANNELS, WV_MAX, WV_MIN

from ChronoEarth.ChronoEarth import ChronoEarth, TemporalChronoEarth, ALL_MEANS, ALL_STDS
from data_utils import pretrain_transform, unimodal_collate_fn
from data_utils.temporal_sampler import BucketBatchSampler, AdaptiveBucketBatchSampler
from data_utils.collate_func import temporal_collate_fn
from data_utils.transforms import pretrain_transform
from torch.utils.data import DataLoader

from GFM_Baselines.models.LESSViT import SpatialSpectralLowRankViTConfig, SpatialSpectralMAEViT
from GFM_Baselines.models.LEASTViT import TemporalLowRankViTConfig, LeastMAEViT
from trainer import MAETrainer, LEASTViTMAETrainer
from args import parse_args
from utils import calculate_modal_loss, get_lasted_checkpoint


logger = get_logger(__name__)

def get_model_and_dataset(args):
    metadata = get_chronoearth_metadata(args.channel_groups)
    optical_mean, optical_std = metadata["mean"], metadata["std"]
    transform = partial(pretrain_transform, optical_mean=optical_mean, optical_std=optical_std, zero_mean=args.zero_mean)

    num_channel_groups = []
    for channel_group in args.channel_groups:
        num_channel_groups.append(NUM_CHANNELS[channel_group])

    if args.model_name == "lessvit":
        model_config = SpatialSpectralLowRankViTConfig(**vars(args), num_channel_groups=num_channel_groups, spatial_resolution=30, input_size=args.crop_size)
        model = SpatialSpectralMAEViT(model_config)

        collate_fn = partial(unimodal_collate_fn, modal=args.modal_mode, transform=None, random_crop=args.random_crop, \
        scale=args.scale, crop_size=args.crop_size, normalize_wv=args.use_rope_embed, wv_max=WV_MAX, wv_min=WV_MIN)

        dataset = dict(train=ChronoEarth(
            split="train",
            cache_dir=args.data_dir,
            regions=args.regions,
            channel_groups=args.channel_groups,
            transform=transform
        ))
    elif args.model_name == "leastvit":
        model_config = TemporalLowRankViTConfig(**vars(args), num_channel_groups=num_channel_groups)
        model = LeastMAEViT(model_config)

        collate_fn = partial(temporal_collate_fn, modal=args.modal_mode, transform=None, random_crop=args.random_crop, \
        scale=args.scale, crop_size=args.crop_size, normalize_wv=args.use_rope_embed, wv_max=WV_MAX, wv_min=WV_MIN)

        dataset = dict(train=TemporalChronoEarth(
            split="train",
            cache_dir=args.data_dir,
            regions=args.regions,
            channel_groups=args.channel_groups,
            num_frames=args.num_frames, 
            frames_lb=args.frames_lb, 
            transform=transform
        ))
    else:
        raise ValueError(f"Expected lessvit or leastvit, but got {args.model_name}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, dataset, collate_fn

def main(args):    
    print(f"Training {args.model_name}")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    assert args.modal_mode == "optical", f"Only optical modal_mode is supported for EO1H-313K dataset, but got {args.modal_mode}"

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
                
    model, dataset, collate_fn = get_model_and_dataset(args)
    custom_loss_function = partial(calculate_modal_loss, loss_type=args.loss_type)
    
    if args.resume_from_checkpoint == "latest":
        args.resume_from_checkpoint = get_lasted_checkpoint(args)
        print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        logging_strategy="steps",
        logging_steps=1,
        ddp_find_unused_parameters=False,
    )
    
    # Set up wandb first if using it
    if args.report_to == "wandb" :
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if training_args.local_rank == 0:
            wandb.init(
                project=f"{args.model_name}-pretrain",
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )
    
    if args.model_name == "leastvit":
        print("use LEASTViTMAETrainer")
        trainer = LEASTViTMAETrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            data_collator=collate_fn,
            compute_loss_func=custom_loss_function,
            modal_mode=args.modal_mode,
        )
    else:
        trainer = MAETrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            data_collator=collate_fn,
            compute_loss_func=custom_loss_function,
            modal_mode=args.modal_mode,
        )
        
    total_batch_size = trainer.args.per_device_train_batch_size * trainer.accelerator.num_processes * trainer.args.gradient_accumulation_steps
    max_steps = trainer.args.max_steps if trainer.args.max_steps != -1 else math.ceil(len(trainer.train_dataset) / total_batch_size) * trainer.args.num_train_epochs

    if training_args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(trainer.train_dataset)}")
        logger.info(f"  Num Epochs = {trainer.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {trainer.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {trainer.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Save final model
    if training_args.local_rank == 0:
        trainer.save_model()
        logger.info("Training completed and model saved.")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)