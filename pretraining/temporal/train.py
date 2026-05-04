import os
import sys
import logging
import math
import random
from functools import partial

from accelerate.logging import get_logger
from transformers import is_wandb_available, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ChronoEarth import get_chronoearth_metadata, NUM_CHANNELS
from ChronoEarth.ChronoEarth import TemporalChronoEarth
from data_utils.transforms import pretrain_transform
from data_utils.temporal_sampler import AdaptiveBucketBatchSampler

# Reuse the proven trainer infrastructure from pretrain_scripts verbatim
from pretrain_scripts.trainer import MAETrainer, LEASTViTMAETrainer

from .args import parse_args
from .temporal_mae import TemporalMAEWrapper
from .collate import temporal_mae_collate_fn

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Encoder factory
# ------------------------------------------------------------------

def build_stage1_encoder(args, channel_wv):
    """
    Returns (encoder, encode_fn, in_chans, encoder_patch_size).
      encode_fn         : (x: B*T, C, H, W) -> (B*T, N+1, D)
      in_chans          : channels for pixel reconstruction target
      encoder_patch_size: spatial patch size the encoder uses
    """
    name = args.stage1_model
    all_in_chans = sum(NUM_CHANNELS[g] for g in args.channel_groups)

    if name == "dofa":
        from GFM_Baselines.wrappers.dofa_wrapper import DOFAEncoder, DOFAConfig
        encoder = DOFAEncoder(DOFAConfig(
            img_size=args.crop_size, patch_size=args.patch_size, embed_dim=args.embed_dim
        ))
        encoder.load_pretrained_weights(args.stage1_checkpoint)
        wave_list = channel_wv.squeeze(0).tolist()
        encode_fn = lambda x: encoder.forward_encoder(x, wave_list=wave_list)
        return encoder, encode_fn, all_in_chans, args.patch_size

    elif name == "satmae":
        from GFM_Baselines.wrappers.satmae_wrapper import SatMAEEncoder, SatMAEConfig
        encoder = SatMAEEncoder(SatMAEConfig(img_size=args.crop_size, embed_dim=args.embed_dim))
        encoder.load_pretrained_weights(args.stage1_checkpoint)
        encode_fn = lambda x: encoder.forward_encoder(x)
        return encoder, encode_fn, all_in_chans, args.patch_size

    elif name == "specvit":
        from GFM_Baselines.wrappers.specvit_wrapper import SpecViTEncoder, SpecViTConfig
        config = SpecViTConfig(embed_dim=args.embed_dim)
        encoder = SpecViTEncoder(config)
        encoder.load_pretrained_weights(args.stage1_checkpoint)
        encode_fn = lambda x: encoder.forward_encoder(x)
        return encoder, encode_fn, all_in_chans, config.token_patch_size

    elif name == "lessvit":
        from GFM_Baselines.wrappers.lessvit_wrapper import LESSViTEncoder, LESSViTConfig
        num_channel_groups = [NUM_CHANNELS[g] for g in args.channel_groups]
        original_num_channel_groups = list(num_channel_groups)
        config = LESSViTConfig(
            patch_size=args.patch_size, embed_dim=args.embed_dim,
            num_patches=(args.crop_size // args.patch_size) ** 2,
            spatial_resolution=30, num_channel_groups=num_channel_groups,
        )
        encoder = LESSViTEncoder(config)
        encoder.load_pretrained_weights(args.stage1_checkpoint)

        def sample_hcs_channels():
            if args.channel_dropout is None:
                return None, original_num_channel_groups
            assert len(args.channel_dropout) in [1, 2], (
                f"channel_dropout should have one or two values, got {args.channel_dropout}"
            )
            channel_dropout = sorted(args.channel_dropout)
            assert 0.0 <= channel_dropout[0] < 1.0 and 0.0 <= channel_dropout[-1] < 1.0, (
                f"channel_dropout should be in [0, 1), got {args.channel_dropout}"
            )

            start = 0
            idx_to_keep = []
            sampled_num_channel_groups = []
            for num_channel in original_num_channel_groups:
                if len(channel_dropout) == 1:
                    keep = max(1, int(num_channel * (1.0 - channel_dropout[0])))
                else:
                    keep_min = max(1, int(num_channel * (1.0 - channel_dropout[1])))
                    keep_max = max(1, int(num_channel * (1.0 - channel_dropout[0])))
                    keep = random.randint(keep_min, keep_max)
                to_keep = sorted(random.sample(range(num_channel), keep))
                idx_to_keep.extend([start + i for i in to_keep])
                sampled_num_channel_groups.append(keep)
                start += num_channel
            return idx_to_keep, sampled_num_channel_groups

        def encode_fn(x):
            idx_to_keep, sampled_num_channel_groups = sample_hcs_channels()
            optical_channel_wv = channel_wv.to(x.device)
            if idx_to_keep is not None:
                x = x[:, idx_to_keep, :, :]
                optical_channel_wv = optical_channel_wv[:, idx_to_keep]
                encoder.encoder.num_channel_groups = sampled_num_channel_groups
            else:
                encoder.encoder.num_channel_groups = original_num_channel_groups

            result = encoder.encoder(
                optical=x,
                optical_channel_wv=optical_channel_wv,
                spatial_resolution=30,
            )
            # channel-CLS row aggregates all spectral bands → (B, HW+1, D)
            return result[0][:, 0, :]
        return encoder, encode_fn, all_in_chans, args.patch_size

    else:
        raise ValueError(f"Unknown stage1_model: {name}")


# ------------------------------------------------------------------
# Temporal-specific trainer — only adds bucket sampler on top of
# LEASTViTMAETrainer which already handles _get_dataloader correctly
# ------------------------------------------------------------------

class TemporalMAETrainer(LEASTViTMAETrainer):
    def __init__(self, max_frames_budget, **kwargs):
        super().__init__(**kwargs)
        self.max_frames_budget = max_frames_budget
        # AdaptiveBucketBatchSampler has no batch_size attribute — disable
        # Accelerate's BatchSamplerShard which requires it for DDP splitting.
        self.accelerator.even_batches = False

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not self._has_length(train_dataset):
            return None
        return AdaptiveBucketBatchSampler(
            lengths=train_dataset.frame_lengths,
            max_frames=self.max_frames_budget,
            boundaries=[1, 2, 4, 6, 8, 12, 16],
            shuffle=True,
            shuffle_buckets=True,
            seed=self.args.seed or 42,
        )


# ------------------------------------------------------------------
# Loss function (mirrors calculate_modal_loss signature)
# ------------------------------------------------------------------

def calculate_temporal_loss(outputs: dict, **kwargs) -> float:
    return outputs["loss"]


# ------------------------------------------------------------------
# Main — mirrors pretrain_scripts/train.py structure exactly
# ------------------------------------------------------------------

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    metadata = get_chronoearth_metadata(args.channel_groups)
    optical_mean, optical_std = metadata["mean"], metadata["std"]
    import torch
    channel_wv = torch.tensor(metadata["channel_wv"]).unsqueeze(0)

    encoder, encode_fn, in_chans, enc_patch_size = build_stage1_encoder(args, channel_wv)
    num_patches = (args.crop_size // enc_patch_size) ** 2

    transform = partial(pretrain_transform, optical_mean=optical_mean,
                        optical_std=optical_std, zero_mean=False)
    dataset = TemporalChronoEarth(
        split="train",
        cache_dir=args.data_dir,
        regions=args.regions,
        channel_groups=args.channel_groups,
        num_frames=args.num_frames,
        frames_lb=args.frames_lb,
        transform=transform,
        slow_sample_threshold=args.slow_sample_threshold,
    )
    collate_fn = temporal_mae_collate_fn

    model = TemporalMAEWrapper(
        encoder=encoder,
        encode_fn=encode_fn,
        channel_wv=channel_wv,
        embed_dim=args.embed_dim,
        num_patches=num_patches,
        patch_size=enc_patch_size,
        in_chans=in_chans,
        n_temporal_layers=args.n_temporal_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_frames=args.num_frames + 4,
        temporal_mask_ratio=args.temporal_mask_ratio,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        norm_pix_loss=args.norm_pix_loss,
        year_range=(args.year_min, args.year_max),
    )

    if args.resume_from_checkpoint == "latest":
        from pretrain_scripts.utils import get_lasted_checkpoint
        args.resume_from_checkpoint = get_lasted_checkpoint(args)
        print(f"Resume from checkpoint: {args.resume_from_checkpoint}")

    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        logging_strategy="steps",
        logging_steps=1,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,   # collate_fn needs num_frames/timestamp from dataset
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use wandb logging.")
        import wandb
        if training_args.local_rank == 0:
            wandb.init(
                project=f"{args.stage1_model}-temporal-pretrain",
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args),
            )

    max_frames_budget = args.max_frames_budget or (args.per_device_train_batch_size * args.num_frames)

    trainer = TemporalMAETrainer(
        max_frames_budget=max_frames_budget,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        compute_loss_func=calculate_temporal_loss,
        modal_mode="optical",
    )

    # Compute steps directly from the sampler — avoids spawning dataloader workers
    # twice (once here, once inside trainer.train()).
    _sampler = AdaptiveBucketBatchSampler(
        lengths=dataset.frame_lengths,
        max_frames=max_frames_budget,
        boundaries=[1, 2, 4, 6, 8, 12, 16],
    )
    steps_per_epoch = len(_sampler)
    max_steps = steps_per_epoch * args.num_train_epochs

    if training_args.local_rank == 0:
        logger.info("***** Running temporal MAE pretraining *****")
        logger.info(f"  Model          = {args.stage1_model}")
        logger.info(f"  Locations      = {len(dataset)}")
        logger.info(f"  Epochs         = {args.num_train_epochs}")
        logger.info(f"  Steps/epoch    = {steps_per_epoch}")
        logger.info(f"  Processes      = {trainer.accelerator.num_processes}")
        logger.info(f"  Total steps    = {max_steps}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if training_args.local_rank == 0:
        trainer.save_model()
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
