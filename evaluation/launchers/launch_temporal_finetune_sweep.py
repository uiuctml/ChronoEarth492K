#!/usr/bin/env python3
"""
Launch finetuning sweeps for stage-2 temporal-pretrained models.

This launcher is intentionally narrower than launch_all_finetune_sweep.py:
- only temporal configs SH, LH, and CD
- only registry entries with temporal_pooling="pretrain"
- pretrained paths default to results/temporal_pretrain/{model_name}_temporal_stage2

Examples:
    python launch_temporal_finetune_sweep.py --dry_run

    python launch_temporal_finetune_sweep.py \
        --temporal_configs LH \
        --datasets NLCDLndCov \
        --models specvit dofa \
        --num_frames 4 8 \
        --learning_rates 3e-5 5e-5
"""

import argparse
import os
import random
import subprocess
import time
from typing import Any, Dict, Optional

from finetune_scripts.registery import MODEL_CONFIGS, TEMPORAL_CONFIGS


def generate_temporal_finetune_command(
    root_dir: str,
    run_name: str,
    dataset_name: str,
    task_type: str,
    temporal_config: str,
    model_config: Dict[str, Any],
    pretrained_model_path: str,
    effective_batch_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    data_dir: Optional[str] = None,
    num_frames: int = 4,
    frames_filter: int = 1,
    num_epochs: int = 40,
    img_size: int = 128,
    crop_size: int = 128,
    scale: float = 1.0,
    ignore_index: int = 255,
    report_to: str = "wandb",
    train_frac: float = 1.0,
    val_frac: float = 1.0,
    zero_mean: bool = False,
    random_crop: bool = False,
    random_rotation: bool = False,
    random_resize: bool = False,
    dataloader_num_workers: int = 16,
) -> str:
    assert model_config.get("temporal_model"), "Temporal launcher only supports temporal models"
    assert model_config.get("temporal_pooling") == "pretrain", "Temporal launcher only supports pretrain pooling"

    physical_batch_size = max(1, batch_size // max(1, num_frames))
    grad_accum_steps = max(1, effective_batch_size // max(1, physical_batch_size))

    cmd = [
        "python finetune_scripts/finetune/finetune.py",
        f"--temporal_config {temporal_config}",
        f"--dataset_name {dataset_name}",
        f"--task_type {task_type}",
        f"--run_name {run_name}",
        f"--data_dir {data_dir or os.path.join(root_dir, 'data')}",
        f"--model_name {model_config['model_name']}",
        f"--pretrained_model_path {pretrained_model_path}",
        f"--decoder_model {model_config['decoder_model']}",
        "--temporal_model",
        f"--num_frames {num_frames}",
        f"--frames_fliter {frames_filter}",
        "--temporal_embedding",
        "--temporal_pooling pretrain",
        f"--per_device_train_batch_size {physical_batch_size}",
        f"--per_device_eval_batch_size {physical_batch_size}",
        f"--gradient_accumulation_steps {grad_accum_steps}",
        f"--num_train_epochs {num_epochs}",
        f"--learning_rate {learning_rate}",
        f"--weight_decay {weight_decay}",
        f"--crop_size {crop_size}",
        f"--img_size {img_size}",
        f"--scale {scale}",
        f"--ignore_index {ignore_index}",
        f"--report_to {report_to}",
        "--lr_scheduler_type cosine",
        "--warmup_ratio 0.05",
        "--max_grad_norm 1.0",
        "--seed 42",
        "--mixed_precision bf16",
        f"--dataloader_num_workers {dataloader_num_workers}",
        "--dataloader_pin_memory",
        "--eval_strategy epoch",
        "--save_strategy epoch",
        "--save_total_limit 1",
        f"--output_dir {root_dir}/results/models",
        f"--logging_dir {root_dir}/results/logs",
        f"--wandb_dir {root_dir}/results/",
        f"--train_frac {train_frac}",
        f"--val_frac {val_frac}",
    ]

    if zero_mean:
        cmd.append("--zero_mean")
    if random_crop:
        cmd.append("--random_crop")
    if random_rotation:
        cmd.append("--random_rotation")
    if random_resize:
        cmd.append("--random_resize")

    return " \\\n    ".join(cmd)


def build_commands(args):
    commands = []
    data_dir = args.data_dir or os.path.join(args.root_dir, "data")

    for temporal_config in args.temporal_configs:
        tasks = TEMPORAL_CONFIGS[temporal_config]
        model_configs = MODEL_CONFIGS[temporal_config]
        num_frames_list = [2] if temporal_config == "CD" else args.num_frames

        for task_type, datasets in tasks.items():
            if task_type not in model_configs:
                continue
            pretrain_model_configs = [
                cfg for cfg in model_configs[task_type]
                if cfg.get("temporal_pooling") == "pretrain"
            ]

            for dataset_name in datasets:
                if args.datasets and dataset_name not in args.datasets:
                    continue

                for model_config in pretrain_model_configs:
                    model_name = model_config["model_name"]
                    if args.models and model_name not in args.models:
                        continue

                    pretrained_path = args.temporal_pretrained_model_path
                    if pretrained_path is None:
                        pretrained_path = os.path.join(
                            args.root_dir,
                            "results",
                            "temporal_pretrain",
                            "{model_name}_temporal_stage2",
                        )
                    pretrained_path = pretrained_path.format(model_name=model_name)

                    for num_frames in num_frames_list:
                        for learning_rate in args.learning_rates:
                            run_name = "_".join([
                                model_name,
                                "pretrainpool",
                                temporal_config,
                                f"T{num_frames}",
                                dataset_name,
                                f"lr{learning_rate:.0e}",
                            ])
                            output_path = os.path.join(
                                args.root_dir,
                                "results",
                                "models",
                                dataset_name,
                                run_name,
                            )
                            test_results_path = os.path.join(output_path, "test_results.json")
                            if args.skip_existing and os.path.exists(test_results_path):
                                print(f"Skipping {run_name} (already exists)")
                                continue

                            cmd = generate_temporal_finetune_command(
                                root_dir=args.root_dir,
                                run_name=run_name,
                                dataset_name=dataset_name,
                                task_type=task_type,
                                temporal_config=temporal_config,
                                model_config=model_config,
                                pretrained_model_path=pretrained_path,
                                effective_batch_size=args.effective_batch_size,
                                batch_size=args.batch_size,
                                learning_rate=learning_rate,
                                weight_decay=args.weight_decay,
                                data_dir=data_dir,
                                num_frames=num_frames,
                                frames_filter=args.frames_filter,
                                num_epochs=args.num_epochs,
                                img_size=args.img_size,
                                crop_size=args.crop_size,
                                scale=args.scale,
                                ignore_index=args.ignore_index,
                                report_to=args.report_to,
                                train_frac=args.train_frac,
                                val_frac=args.val_frac,
                                zero_mean=args.zero_mean,
                                random_crop=args.random_crop,
                                random_rotation=args.random_rotation,
                                random_resize=args.random_resize,
                                dataloader_num_workers=args.dataloader_num_workers,
                            )
                            commands.append((run_name, cmd, output_path))

    return commands


def save_commands(commands):
    for _, cmd, output_path in commands:
        os.makedirs(output_path, exist_ok=True)
        cmd_file = os.path.join(output_path, "launch_command.sh")
        with open(cmd_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)
            f.write("\n")
        os.chmod(cmd_file, 0o755)


def local_launcher(commands):
    print(f"Launching {len(commands)} jobs serially")
    for i, (run_name, cmd, _) in enumerate(commands):
        print(f"[{i + 1}/{len(commands)}] Running {run_name}")
        print(f"Command: {cmd[:160]}...")
        result = subprocess.call(cmd, shell=True)
        if result != 0:
            print(f"Warning: Command for {run_name} exited with code {result}")
        else:
            print(f"Completed {run_name}")


def multi_gpu_launcher(commands, available_gpus, max_parallel):
    print(f"Launching {len(commands)} jobs on {len(available_gpus)} GPUs, max_parallel={max_parallel}")
    procs_by_gpu = [None] * len(available_gpus)
    command_queue = list(commands)
    completed = 0

    while command_queue or any(proc is not None for proc in procs_by_gpu):
        for idx, gpu_id in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if proc is not None and proc.poll() is not None:
                completed += 1
                print(f"Completed {completed}/{len(commands)} jobs")
                procs_by_gpu[idx] = None

            if procs_by_gpu[idx] is None and command_queue:
                run_name, cmd, _ = command_queue.pop(0)
                print(f"[{completed}/{len(commands)}] Launching {run_name} on GPU {gpu_id}")
                procs_by_gpu[idx] = subprocess.Popen(
                    f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}",
                    shell=True,
                )

        time.sleep(2)

    print(f"All {len(commands)} jobs completed")


def main(system_args=None):
    parser = argparse.ArgumentParser(description="Launch temporal-pretrained finetune sweeps")
    parser.add_argument("--root_dir", type=str, default=os.getcwd())
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--gpu_devices", type=str, default="0")
    parser.add_argument("--temporal_pretrained_model_path", type=str, default=None,
                        help="Can use {model_name}. Default: {root_dir}/results/temporal_pretrain/{model_name}_temporal_stage2")
    parser.add_argument("--temporal_configs", type=str, nargs="+", default=["SH", "LH", "CD"],
                        choices=["SH", "LH", "CD"])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Subset of temporal-pretrained backbones, e.g. specvit dofa satmae lessvit")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[3e-5, 5e-5, 8e-5, 1e-4])
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Nominal batch size before temporal per-frame adjustment")
    parser.add_argument("--effective_batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_frames", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--frames_filter", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=1.2)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--zero_mean", action="store_true")
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--random_rotation", action="store_true")
    parser.add_argument("--random_resize", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_commands", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=None)

    args = parser.parse_args(system_args)

    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{args.root_dir}"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    available_gpus = [gpu for gpu in args.gpu_devices.split(",") if gpu]
    max_parallel = args.max_parallel or len(available_gpus)

    commands = build_commands(args)
    print(f"\nGenerated {len(commands)} temporal-pretrained finetune commands")
    print(f"Will use {max_parallel} parallel jobs\n")

    if args.dry_run:
        for run_name, cmd, _ in commands[:20]:
            print(f"Run: {run_name}")
            print(f"Command:\n{cmd}\n")
        if len(commands) > 20:
            print(f"... and {len(commands) - 20} more commands")
        return

    if args.save_commands:
        save_commands(commands)
        print(f"Saved {len(commands)} launch commands")

    if max_parallel > 1 and len(available_gpus) > 1:
        multi_gpu_launcher(commands, available_gpus, max_parallel)
    else:
        local_launcher(commands)


if __name__ == "__main__":
    main(None)
