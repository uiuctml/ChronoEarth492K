# --------------------------------------------------------
# SatMAE Finetune Main Script
# Based on: https://github.com/sustainlab-group/SatMAE
# --------------------------------------------------------

import os
import sys
import torch
import logging
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import optuna
from transformers.integrations import is_wandb_available

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from args import parse_args
from utils import (
    get_task_model, get_loss_fn, get_metric, 
)
# from benchmark.hsi.GFMBench.utils import get_dataset_infos, load_dataset, get_metadata
from ChronoEarth import StaticTask, ALL_BANDS, ALL_BANDS_MEAN, ALL_BANDS_STD, get_downstream_dataset, NUM_CLASSES
from data_utils import downstream_collate_fn, segmentation_transform_one_sample, downstream_collate_fn, get_downstream_transform
from functools import partial

def model_init_template(trial):
    args = parse_args()
    model = get_task_model(args, NUM_CLASSES[args.dataset_name])
    return model

def main(args):    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    if args.logging_dir is not None:
        os.makedirs(args.logging_dir, exist_ok=True)

    # Load dataset
    # args.data_dir = "./data"  # Use a local data path for debugging.
    # Ensure crop_size aligns with model input size if not explicitly set
    if getattr(args, 'crop_size', None) is None:
        args.crop_size = getattr(args, 'img_size', None)
        print(f"crop size is not specified, set to img_size: {args.crop_size}")
    
    optical_mean, optical_std = ALL_BANDS_MEAN, ALL_BANDS_STD
    radar_mean, radar_std = None, None # to supress error msg
        
    train_transform, eval_transform = get_downstream_transform(args.task_type, args.crop_size, args.scale, args.random_rotation,
                                                    optical_mean, optical_std, radar_mean, radar_std,
                                                    random_crop=args.random_crop, random_resize=args.random_resize, zero_mean=args.zero_mean)
    
    dataset_dict = get_downstream_dataset(args, train_transform, eval_transform)
    
    model_init = partial(model_init_template)
    collate_fn = partial(
        downstream_collate_fn,
        scale=args.scale,
        crop_size=args.crop_size,
        normalize_wv=(args.model_name=="lessvit") # only less vit needs normalization of channel wv so far
    )
    
    # get loss function and metric
    ignore_index = getattr(args, 'ignore_index', -1)
    custom_loss_function = get_loss_fn(args.task_type, ignore_index=ignore_index, binary_label=NUM_CLASSES[args.dataset_name]==1)
    compute_metrics, metric_name = get_metric(args.task_type, NUM_CLASSES[args.dataset_name], ignore_index=ignore_index) 

    if 'evaluation_strategy' in TrainingArguments.__dataclass_fields__:
        eval_strategy_value = getattr(args, 'eval_strategy', getattr(args, 'evaluation_strategy', 'no'))
        setattr(args, 'eval_strategy', eval_strategy_value)
        if 'save_strategy' in TrainingArguments.__dataclass_fields__ and not getattr(args, 'lp', False):
            # If we intend to load best model at end, match save with eval to satisfy HF requirement
            setattr(args, 'save_strategy', eval_strategy_value)


    # Create TrainingArguments with evaluation settings
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        full_determinism=False,
        # dispatch_batches=None,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        load_best_model_at_end=True if not args.lp else False,
        greater_is_better=False if args.task_type == "regression" else True,
        logging_strategy="steps" if not args.lp else "epoch",
        logging_steps=1 if not args.lp else None,
        logging_first_step=True,
        metric_for_best_model=metric_name,
        remove_unused_columns=False,
        label_names=["labels"]
    )
    
    callbacks = []
    # if args.use_early_stopping and not args.lp:
    #     callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold))
    
    trainer = Trainer(
        model=model_init(None),
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['val'],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        compute_loss_func=custom_loss_function,
    )
    
    # Set up wandb first if using it
    if args.report_to == "wandb" :
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if training_args.local_rank == 0:
            wandb.init(
                project=f"chronoearth-{args.dataset_name}",
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )
    
    # Train the model with best hyperparameters
    train_result = trainer.train()
    
    # Final evaluation
    metrics = trainer.evaluate(eval_dataset=dataset_dict['test'])
    
    # Log the metrics
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    
    # # Final evaluation
    # metrics = trainer.evaluate(eval_dataset=dataset_dict['val'])
    
    # # Log the metrics
    # trainer.log_metrics("val", metrics)
    # trainer.save_metrics("val", metrics)
    
    # if ood_test is present, evaluate on it
    for key in dataset_dict.keys():
        if 'ood' in key:
            metrics = trainer.evaluate(eval_dataset=dataset_dict[key])
            trainer.log_metrics(key, metrics)
            trainer.save_metrics(key, metrics)
    
    # # Save the final model
    # trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    # Save training state
    trainer.save_state()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
