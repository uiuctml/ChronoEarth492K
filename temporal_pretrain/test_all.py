"""
Smoke-test for all temporal MAE baselines.

Runs 10 training epochs with minimal settings for each of:
  dofa / satmae / specvit / lessvit

Usage (from repo root):
  python -m temporal_pretrain.test_all \
      --data_dir /path/to/EO1H \
      --model_dir ./checkpoints/baseline_models \
      --output_dir /tmp/temporal_test
"""

import sys
import os
import math
import time
import traceback
import argparse
from functools import partial
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import seed_worker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ChronoEarth import get_chronoearth_metadata, NUM_CHANNELS
from ChronoEarth.ChronoEarth import TemporalChronoEarth
from data_utils.transforms import pretrain_transform
from data_utils.temporal_sampler import AdaptiveBucketBatchSampler

from temporal_pretrain.temporal_mae import TemporalMAEWrapper
from temporal_pretrain.collate import temporal_mae_collate_fn
from temporal_pretrain.train import build_stage1_encoder, TemporalMAETrainer


# ------------------------------------------------------------------
# Test configuration — kept small so the whole suite runs quickly
# ------------------------------------------------------------------

TEST_EPOCHS       = 1
BATCH_BUDGET      = 16      # max_frames_budget for AdaptiveBucketBatchSampler
FRAMES_LB         = 2       # only locations with ≥ 2 frames
NUM_FRAMES        = 4       # cap per location
CROP_SIZE         = 128
PATCH_SIZE        = 16
EMBED_DIM         = 768
N_TEMPORAL_LAYERS = 2       # shallow — just to test the forward pass
REGION            = ["AC"]  # one region for speed
CHANNEL_GROUPS    = ["VNIR", "SWIR1", "SWIR2", "SWIR3", "SWIR4"]

CHECKPOINT_NAMES = {
    # "dofa":     "DOFA",
    # "satmae":   "SatMAE",
    # "specvit":  "SpectralViT",
    "lessvit":  "LESSViT/LESSVIT_b2_d8_r1",
}


# ------------------------------------------------------------------
# Minimal trainer that reuses our production TemporalMAETrainer logic
# ------------------------------------------------------------------

class TestTrainer(TemporalMAETrainer):
    """Thin subclass — reuses TemporalMAETrainer's _get_dataloader/_get_train_sampler."""
    pass


# ------------------------------------------------------------------
# Per-model test
# ------------------------------------------------------------------

@dataclass
class TestResult:
    model_name: str
    passed: bool
    elapsed: float
    final_loss: float | None
    error: str | None


def run_one(model_name: str, model_dir: str, data_dir: str, output_dir: str) -> TestResult:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Testing: {model_name.upper()}")
    print(f"{'='*60}")

    try:
        # ---- Dataset ----
        metadata = get_chronoearth_metadata(CHANNEL_GROUPS)
        optical_mean, optical_std = metadata["mean"], metadata["std"]
        channel_wv = torch.tensor(metadata["channel_wv"]).unsqueeze(0)   # (1, C)

        transform = partial(pretrain_transform,
                            optical_mean=optical_mean,
                            optical_std=optical_std,
                            zero_mean=False)

        dataset = TemporalChronoEarth(
            split="train",
            cache_dir=data_dir,
            regions=REGION,
            channel_groups=CHANNEL_GROUPS,
            num_frames=NUM_FRAMES,
            frames_lb=FRAMES_LB,
            transform=transform,
        )
        print(f"  Dataset: {len(dataset)} locations, "
              f"frame lengths: min={dataset.frame_lengths.min()} "
              f"max={dataset.frame_lengths.max()}")

        # ---- Build encoder ----

        class _Args:
            stage1_model    = model_name
            stage1_checkpoint = os.path.join(model_dir, CHECKPOINT_NAMES[model_name])
            channel_groups  = CHANNEL_GROUPS
            crop_size       = CROP_SIZE
            patch_size      = PATCH_SIZE
            embed_dim       = EMBED_DIM

        encoder, encode_fn, in_chans, enc_patch_size = build_stage1_encoder(_Args(), channel_wv)
        print(f"  Encoder loaded. in_chans={in_chans}, patch_size={enc_patch_size}")

        # ---- Quick forward-pass smoke test before training ----
        num_patches = (CROP_SIZE // enc_patch_size) ** 2
        model = TemporalMAEWrapper(
            encoder=encoder,
            encode_fn=encode_fn,
            channel_wv=channel_wv,
            embed_dim=EMBED_DIM,
            num_patches=num_patches,
            patch_size=enc_patch_size,
            in_chans=in_chans,
            n_temporal_layers=N_TEMPORAL_LAYERS,
            num_heads=12,
            mlp_ratio=4.0,
            max_frames=NUM_FRAMES + 2,
            temporal_mask_ratio=0.5,
            decoder_embed_dim=256,
            norm_pix_loss=True,
        )

        # Single-batch forward pass
        collate_fn = temporal_mae_collate_fn
        sampler = AdaptiveBucketBatchSampler(
            lengths=dataset.frame_lengths,
            max_frames=BATCH_BUDGET,
            boundaries=[1, 2, 4],
            shuffle=False,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(
                optical=batch["optical"],
                timestamps=batch["timestamps"],
                valid_mask=batch["valid_mask"],
            )
        print(f"  Forward pass OK. loss={out['loss'].item():.4f}, "
              f"masked frames={out['n_masked_frames'].item()}")

        # ---- training ----
        from temporal_pretrain.train import calculate_temporal_loss
        run_output = os.path.join(output_dir, model_name)
        training_args = TrainingArguments(
            output_dir=run_output,
            per_device_train_batch_size=4,
            num_train_epochs=TEST_EPOCHS,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=(not torch.cuda.is_bf16_supported() and torch.cuda.is_available()),
            save_strategy="no",
            report_to="none",
            logging_strategy="epoch",
            seed=42,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        )

        trainer = TestTrainer(
            max_frames_budget=BATCH_BUDGET,
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn,
            compute_loss_func=calculate_temporal_loss,
            modal_mode="optical",
        )

        train_result = trainer.train()
        final_loss = train_result.training_loss
        elapsed = time.time() - t0

        print(f"  PASSED in {elapsed:.1f}s — final loss: {final_loss:.4f}")
        return TestResult(model_name, True, elapsed, final_loss, None)

    except Exception:
        elapsed = time.time() - t0
        err = traceback.format_exc()
        print(f"  FAILED after {elapsed:.1f}s")
        print(err)
        return TestResult(model_name, False, elapsed, None, err)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default=os.path.join(os.getcwd(), "data", "EO1H"))
    parser.add_argument("--model_dir",  type=str,
                        default=os.path.join(os.getcwd(), "checkpoints", "baseline_models"))
    parser.add_argument("--output_dir", type=str,
                        default="/tmp/temporal_test")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["dofa", "satmae", "specvit", "lessvit"],
                        choices=["dofa", "satmae", "specvit", "lessvit"],
                        help="Subset of models to test")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for model_name in args.models:
        result = run_one(model_name, args.model_dir, args.data_dir, args.output_dir)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        loss_str = f"loss={r.final_loss:.4f}" if r.final_loss is not None else "no loss"
        print(f"  [{status}] {r.model_name:<10}  {r.elapsed:>6.1f}s  {loss_str}")
        if not r.passed:
            all_passed = False

    print(f"{'='*60}")
    if all_passed:
        print("  All baselines passed.")
    else:
        print("  Some baselines FAILED — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
