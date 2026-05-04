#!/usr/bin/env python3
"""
Comprehensive finetuning launch script for all datasets and models with hyperparameter search.

This script:
- Iterates over all temporal configs (S, SH, LH, CD)
- Iterates over all datasets for each temporal config
- Iterates over all models for each task type
- Performs hyperparameter search (learning rate, batch size, weight decay)
- Handles GPU allocation and parallel execution
- Saves commands to files for reproducibility

Usage examples:
    # Dry run to see what would be executed
    python launch_all_finetune_sweep.py --dry_run
    
    # Run all experiments with default hyperparameters
    python launch_all_finetune_sweep.py --pretrained_model_path /path/to/models/{model_name}/checkpoint-xxx/model.safetensors
    
    # Run specific datasets and models
    python launch_all_finetune_sweep.py --datasets CLCD CDL --models satmae specvit
    
    # Custom hyperparameter search
    python launch_all_finetune_sweep.py --learning_rates 1e-4 3e-4 5e-4 --batch_sizes 8 16 --weight_decays 0.01 0.05
    
    # Skip existing runs
    python launch_all_finetune_sweep.py --skip_existing --save_commands
"""

import argparse
import os
import subprocess
import random
import time
from typing import Optional, Dict, Any

# Import registry configurations
import sys
sys.path.append(os.path.dirname(__file__))
from finetune_scripts.registery import (
    TEMPORAL_CONFIGS, 
    MODEL_CONFIGS, 
    MODEL_NAMES_MAP,
)

def generate_finetune_command(
    root_dir: str,
    run_name: str,
    dataset_name: str,
    task_type: str,
    temporal_config: str,
    effective_batch_size: int,
    model_config: Dict[str, Any],
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    port: int,
    pretrained_model_path: str,
    n_gpus: int = 1,
    accelerator_config: str = "",
    data_dir: Optional[str] = None,
    num_frames: int = 4,
    frames_fliter: int = 1,
    num_epochs: int = 10,
    img_size: int = 128,
    ignore_index: int = 255,
    report_to: str = "wandb",
    train_frac: float = 1.0,
    val_frac: float = 1.0,
    zero_mean: bool = False,
    random_crop: bool = False,
    random_rotation: bool = False,
    random_resize: bool = False,
    crop_size: int = 128,
    scale: int = 1,
) -> str:
    """
    Generate a finetuning command string.
    
    Args:
        root_dir: Root directory of the project
        run_name: Unique run name
        temporal_config: Temporal configuration (S, SH, LH, CD)
        model_config: Model configuration dictionary
        learning_rate: Learning rate
        batch_size: Per device batch size
        weight_decay: Weight decay
        port: Port for distributed training
        pretrained_model_path: Path to pretrained model
        n_gpus: Number of GPUs
        accelerator_config: Accelerate config file path
        data_dir: Data directory path
    
    Returns:
        Command string
    """
    script = "python finetune_scripts/finetune/finetune.py"
    
    # Calculate gradient accumulation steps
    grad_accum_steps = max(1, effective_batch_size // n_gpus // batch_size)
    
    # cmd = [
    #     "accelerate launch",
    #     f"--main_process_port {port}"
    # ]
    
    # if accelerator_config:
    #     cmd.append(f"--config_file {accelerator_config}")
    # elif n_gpus > 1:
    #     cmd.append(f"--num_processes {n_gpus}")
    cmd = []
    
    cmd.append(script)
    
    # Required arguments
    cmd.extend([
        f"--temporal_config {temporal_config}",
        f"--dataset_name {dataset_name}",
        f"--task_type {task_type}",
        f"--run_name {run_name}",
    ])
    
    # Data arguments
    if data_dir:
        cmd.append(f"--data_dir {data_dir}")
    else:
        cmd.append(f"--data_dir {root_dir}/data")
    
    # Model arguments
    cmd.append(f"--model_name {model_config['model_name']}")
    if pretrained_model_path:
        cmd.append(f"--pretrained_model_path {pretrained_model_path}")
    
    if model_config.get('decoder_model') and model_config.get('decoder_model') != "none":
        cmd.append(f"--decoder_model {model_config['decoder_model']}")
    
    if model_config.get('temporal_model'):
        cmd.append("--temporal_model")
        batch_size = batch_size // num_frames
        grad_accum_steps = grad_accum_steps * num_frames
        cmd.extend([
            f"--num_frames {num_frames}",
            f"--frames_fliter {frames_fliter}",
        ])
    
    if model_config.get('temporal_embedding'):
        cmd.append("--temporal_embedding")

    if zero_mean:
        cmd.append("--zero_mean")
    
    if model_config.get('temporal_pooling'):
        cmd.append(f"--temporal_pooling {model_config['temporal_pooling']}")
    
    if random_crop:
        cmd.append("--random_crop")
    if random_rotation:
        cmd.append("--random_rotation")
    
    # Training arguments
    cmd.extend([
        f"--per_device_train_batch_size {batch_size}",
        f"--per_device_eval_batch_size {batch_size}",
        f"--gradient_accumulation_steps {grad_accum_steps}",
        f"--num_train_epochs {num_epochs}",
        f"--learning_rate {learning_rate}",
        f"--weight_decay {weight_decay}",
        f"--crop_size {crop_size}",
        f"--img_size {img_size}",
        f"--scale {scale}",
        f"--ignore_index {ignore_index}",
        f"--report_to {report_to}",
    ])
    
    # Default training arguments
    cmd.extend([
        "--lr_scheduler_type cosine",
        "--warmup_ratio 0.05",
        "--max_grad_norm 1.0",
        "--seed 42",
        "--mixed_precision bf16",
        "--dataloader_num_workers 16",
        "--dataloader_pin_memory",
        "--eval_strategy epoch",
        "--save_strategy epoch",
        "--save_total_limit 1",
    ])
    
    # Output directories
    cmd.extend([
        f"--output_dir {root_dir}/results/models",
        f"--logging_dir {root_dir}/results/logs",
        f"--wandb_dir {root_dir}/results/",
    ])
    
    # Dataset-specific arguments
    if train_frac:
        cmd.append(f"--train_frac {train_frac}")
    if val_frac:
        cmd.append(f"--val_frac {val_frac}")
    
    return " \\\n    ".join(cmd)

def main(system_args):
    parser = argparse.ArgumentParser(
        description="Launch finetuning for all datasets and models with hyperparameter search"
    )
    parser.add_argument("--root_dir", type=str, default=os.getcwd(),
                       help="Root directory of the project")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory (default: {root_dir}/data)")
    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3",
                       help="Comma-separated GPU device IDs")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                       help="Path to pretrained model (can use {model_name} placeholder)")
    parser.add_argument("--temporal_pretrained_model_path", type=str, default=None,
                       help="Path to temporal-pretrained model for temporal_pooling=pretrain "
                            "(can use {model_name} placeholder). Defaults to "
                            "{root_dir}/results/temporal_pretrain/{model_name}_temporal_stage2")
    parser.add_argument("--temporal_configs", type=str, nargs="+", 
                       default=["S", "SH", "LH", "CD"],
                       choices=["S", "SH", "LH", "CD"],
                       help="Temporal configurations to run")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Specific datasets to run (default: all)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="Specific models to run (default: all)")
    parser.add_argument("--learning_rates", type=float, nargs="+",
                       default=[1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4],
                       help="Learning rates to search")
    parser.add_argument("--batch_size", type=int,
                       default=16,
                       help="Batch size")
    parser.add_argument("--weight_decay", type=float,
                       default=0.01,
                       help="Weight decay")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip runs that already have test_results.json")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only generate commands, don't execute")
    parser.add_argument("--save_commands", action="store_true",
                       help="Save commands to files")
    parser.add_argument("--max_parallel", type=int, default=None,
                       help="Maximum number of parallel jobs (default: number of GPUs)")
    parser.add_argument("--accelerator_config", type=str, default="",
                       help="Path to accelerate config file")
    parser.add_argument("--effective_batch_size", type=int, default=128,
                       help="Effective batch size")
    parser.add_argument("--zero_mean", action="store_true",
                       help="Whether to use zero mean for training")
    parser.add_argument("--num_epochs", type=int, default=40,
                       help="Number of epochs")
    parser.add_argument("--num_frames", type=int, default=[1], nargs="+",
                       help="Number of frames used in temporal model")
    parser.add_argument("--random_crop", action="store_true",
                       help="Whether to use random crop for training")
    parser.add_argument("--random_rotation", action="store_true",
                       help="Whether to use random rotation for training")
    parser.add_argument("--random_resize", action="store_true",
                       help="Whether to use random resize for training")
    parser.add_argument("--scale", type=float, default=1.5,
                       help="Scale of the model")
    parser.add_argument("--crop_size", type=int, default=128,
                       help="Crop size of the model")
    
    args = parser.parse_args(system_args)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{args.root_dir}"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    
    # Determine number of GPUs
    available_gpus = [x for x in args.gpu_devices.split(',') if x != '']
    n_gpus = len(available_gpus)
    max_parallel = args.max_parallel if args.max_parallel else n_gpus
    
    # Set data directory
    data_dir = args.data_dir if args.data_dir else os.path.join(args.root_dir, "data")
    
    # Generate all commands
    command_list = []
    total_combinations = 0
    
    for temporal_config in args.temporal_configs:
        if temporal_config not in TEMPORAL_CONFIGS:
            print(f"Warning: Temporal config {temporal_config} not found, skipping")
            continue
        
        tasks = TEMPORAL_CONFIGS[temporal_config]
        model_configs = MODEL_CONFIGS[temporal_config]
        
        if temporal_config in ["S"]:
            num_frames = [1]
        elif temporal_config in ['CD']:
            num_frames = [2]
        else:
            num_frames = args.num_frames
        
        for task_type, datasets in tasks.items():
            if args.datasets and not any(d in args.datasets for d in datasets):
                continue
            
            if task_type not in model_configs:
                print(f"Warning: No models for task type {task_type} in {temporal_config}, skipping")
                continue
            
            models = model_configs[task_type]
            
            for dataset_name in datasets:
                if args.datasets and dataset_name not in args.datasets:
                    continue
                
                for model_config in models:
                    model_name = model_config['model_name']
                    
                    if args.models and model_name not in args.models:
                        continue
                    
                    # Generate pretrained model path if not provided
                    pretrained_path = args.pretrained_model_path
                    if model_config.get("temporal_pooling") == "pretrain":
                        pretrained_path = (
                            args.temporal_pretrained_model_path
                            or os.path.join(args.root_dir, "results", "temporal_pretrain", "{model_name}_temporal_stage2")
                        )
                        if pretrained_path and "{model_name}" in pretrained_path:
                            pretrained_path = pretrained_path.format(model_name=model_name)
                    elif pretrained_path and "{model_name}" in pretrained_path:
                        model_dir_name = MODEL_NAMES_MAP[model_name]
                        pretrained_path = pretrained_path.format(model_name=model_dir_name)
                        
                    for num_frame in num_frames:
                    
                        # Hyperparameter search
                        for lr in args.learning_rates:
                            # Generate run name
                            run_name_parts = [
                                model_name,
                                temporal_config,
                                dataset_name,
                                f"lr{lr:.0e}",
                            ]
                            
                            if model_config.get('temporal_pooling') and model_config.get('temporal_model'):
                                run_name_parts.insert(-3, f"{model_config['temporal_pooling']}pool")
                                run_name_parts.insert(-2, f"T{num_frame}")
                            
                            run_name = "_".join(run_name_parts)
                            
                            # Check if already exists
                            output_path = os.path.join(
                                args.root_dir, "results", "models", dataset_name, run_name
                            )
                            test_results_path = os.path.join(output_path, "test_results.json")
                            
                            if args.skip_existing and os.path.exists(test_results_path):
                                print(f"Skipping {run_name} (already exists)")
                                continue
                            
                            # Generate port
                            port = random.randint(10000, 65535)
                            
                            # Generate command
                            cmd = generate_finetune_command(
                                root_dir=args.root_dir,
                                run_name=run_name,
                                temporal_config=temporal_config,
                                dataset_name=dataset_name,
                                task_type=task_type,
                                effective_batch_size=args.effective_batch_size,
                                model_config=model_config,
                                learning_rate=lr,
                                batch_size=args.batch_size,
                                weight_decay=args.weight_decay,
                                port=port,
                                pretrained_model_path=pretrained_path,
                                n_gpus=1,
                                accelerator_config=args.accelerator_config,
                                data_dir=data_dir,
                                zero_mean=args.zero_mean,
                                num_epochs=args.num_epochs,
                                ignore_index=255,
                                num_frames=num_frame,
                                random_crop=args.random_crop,
                                random_rotation=args.random_rotation,
                                random_resize=args.random_resize,
                                scale=args.scale,
                                crop_size=args.crop_size,
                            )
                        
                            command_list.append((run_name, cmd, output_path))
                            total_combinations += 1
    
    print(f"\nGenerated {total_combinations} finetuning commands")
    print(f"Will use {max_parallel} parallel jobs\n")
    
    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")
        for run_name, cmd, _ in command_list[:10]:  # Show first 10
            print(f"Run: {run_name}")
            print(f"Command:\n{cmd}\n")
        if len(command_list) > 10:
            print(f"... and {len(command_list) - 10} more commands\n")
        return
    
    # Save commands if requested
    if args.save_commands:
        commands_dir = os.path.join(args.root_dir, "results", "commands")
        os.makedirs(commands_dir, exist_ok=True)
        
        for run_name, cmd, output_path in command_list:
            os.makedirs(output_path, exist_ok=True)
            cmd_file = os.path.join(output_path, "launch_command.sh")
            with open(cmd_file, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(cmd)
            os.chmod(cmd_file, 0o755)
        
        print(f"Saved {len(command_list)} commands to {commands_dir}")
    
    # Execute commands
    if max_parallel > 1 and n_gpus > 1:
        multi_gpu_launcher(command_list, available_gpus, max_parallel)
    else:
        local_launcher(command_list)

def multi_gpu_launcher(commands, available_gpus, max_parallel):
    """
    Launch commands on multiple GPUs in parallel.
    """
    print(f'Launching {len(commands)} jobs on {len(available_gpus)} GPUs (max {max_parallel} parallel)')
    
    procs_by_gpu = [None] * len(available_gpus)
    command_queue = list(commands)
    completed = 0
    
    while command_queue or any(p is not None for p in procs_by_gpu):
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            
            # Check if process finished
            if proc is not None:
                if proc.poll() is not None:
                    # Process finished
                    completed += 1
                    print(f"Completed {completed}/{len(commands)} jobs")
                    procs_by_gpu[idx] = None
            
            # Launch new command if GPU is free and we have commands
            if procs_by_gpu[idx] is None and command_queue:
                run_name, cmd, _ = command_queue.pop(0)
                print(f"[{completed}/{len(commands)}] Launching {run_name} on GPU {gpu_idx}")
                print(f"Command: {cmd[:100]}...")
                
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                procs_by_gpu[idx] = new_proc
        
        time.sleep(2)  # Check every 2 seconds
    
    # Wait for remaining processes
    for p in procs_by_gpu:
        if p is not None:
            p.wait()
    
    print(f"\nAll {len(commands)} jobs completed!")

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    print(f'Launching {len(commands)} jobs serially')
    
    for i, (run_name, cmd, _) in enumerate(commands):
        print(f"[{i+1}/{len(commands)}] Running {run_name}")
        print(f"Command: {cmd[:100]}...")
        
        result = subprocess.call(cmd, shell=True)
        
        if result != 0:
            print(f"Warning: Command for {run_name} exited with code {result}")
        else:
            print(f"Completed {run_name}")

if __name__ == "__main__":
    main(None)
