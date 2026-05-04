import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from typing import Dict
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, average_precision_score, jaccard_score, f1_score, confusion_matrix
from GFM_Baselines.downstream_models import (
    TaskModelConfig, 
    TaskModel,
    PretrainedTemporalTaskModel,
)
from GFM_Baselines.registery import ENCODER_CONFIGS, ENCODER_MODELS

# Supported model names - all use the same baseline architecture
SUPPORTED_MODELS = ENCODER_CONFIGS.keys()

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight=10):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
    
    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight.to(input.device))

def get_task_model(args, num_labels=None):
    if args.model_name not in SUPPORTED_MODELS:
        raise NotImplementedError(f"Model {args.model_name} not supported. Available models: {SUPPORTED_MODELS}")
    
    image_size = args.crop_size if args.crop_size is not None else getattr(args, 'img_size', None)
    args.image_size = image_size
    pretrained_model_path = getattr(args, 'pretrained_model_path', None)
    
    config = TaskModelConfig(**vars(args), num_labels=num_labels)
    if getattr(args, "temporal_pooling", None) == "pretrain":
        model = PretrainedTemporalTaskModel(config)
    else:
        model = TaskModel(config)
    if pretrained_model_path:
        model.load_pretrained_encoder(pretrained_model_path)
    return model


def custom_loss_function(outputs, labels, num_items_in_batch, loss_fct, weight=None):
    """
    Custom loss function.
    Modify this function based on your specific task.
    """
    logits = outputs.get("logits")
    # print(torch.unique(labels))
    if isinstance(loss_fct, torch.nn.CrossEntropyLoss):
        loss = loss_fct(logits.to(torch.float32), labels.to(torch.long))
    else:
        loss = loss_fct(logits.to(torch.float32), labels.to(torch.float32))
    return loss

def get_loss_fn(task_type, ignore_index=255, weights=None, binary_label=False):
    if task_type == "classification" or task_type == "segmentation":
        if binary_label:
            loss_fct = WeightedBCEWithLogitsLoss(pos_weight=10)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)
    elif task_type == "multilabel":
        loss_fct = torch.nn.MultiLabelSoftMarginLoss()
    elif task_type == "regression":
        loss_fct = torch.nn.MSELoss()
    else:
        raise NotImplementedError
    
    loss_fn = partial(custom_loss_function, loss_fct=loss_fct, weight=weights)
    return loss_fn

def compute_metrics_acc(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    f1 = f1_score(labels.flatten(), predictions.flatten(), average="weighted")

    return {"accuracy": accuracy, "f1": f1}

def compute_metrics_mAP(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    valid_mask = labels.sum(axis=0) > 0

    per_class_AP = average_precision_score(labels, predictions, average=None)
    macro_mAP = np.mean(per_class_AP[valid_mask])
    micro_mAP = average_precision_score(labels, predictions, average="micro")

    return {"macro_mAP": macro_mAP, "micro_mAP": micro_mAP}
    # return {"micro_mAP": micro_mAP}

def compute_metrics_IoU(eval_pred: EvalPrediction, ignore_index=255, num_classes=11) -> Dict:
    assert num_classes > 1, "Use compute_metrics_IoU_binary for binary classification"
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(predictions, axis=1)
    
    print(f"raw predictions unique: {np.unique(predictions)}, raw labels unique: {np.unique(labels)}")

    # Flatten for metrics calculation
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    # Mask out ignore_index
    mask = labels_flat != ignore_index
    predictions_flat = predictions_flat[mask]
    labels_flat = labels_flat[mask]
    
    labeles_unique = np.unique(labels_flat)
    print(f"predictions unique: {np.unique(predictions_flat)}, labels unique: {labeles_unique}")
    
    # Calculate IoU
    IoU = jaccard_score(labels_flat, predictions_flat, labels=labeles_unique, average="macro")
    
    # Calculate F1 score
    f1 = f1_score(labels_flat, predictions_flat, average="weighted")
    
    # Calculate accuracy
    accuracy = accuracy_score(labels_flat, predictions_flat)
    
    return {"IoU": IoU, "f1": f1, "accuracy": accuracy}

def compute_metrics_IoU_binary(eval_pred: EvalPrediction, ignore_index=0, num_classes=1) -> Dict:
    assert num_classes == 1, "Use compute_metrics_IoU for multi-class classification"
    assert ignore_index == 0, "Background class must be 0 for binary classification"
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = torch.sigmoid(torch.tensor(predictions))
    predictions = (probs > 0.5).numpy().astype(np.int64)
    
    print(f"raw predictions unique: {np.unique(predictions)}, raw labels unique: {np.unique(labels)}")

    # Flatten for metrics calculation
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    labeles_unique = np.unique(labels_flat)
    print(f"predictions unique: {np.unique(predictions_flat)}, labels unique: {labeles_unique}")
    
    # Foreground-focused metrics (positive class = 1)
    IoU = jaccard_score(labels_flat, predictions_flat, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(labels_flat, predictions_flat, average="binary", pos_label=1, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels_flat, predictions_flat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {"IoU": IoU, "f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}

def compute_metrics_rmse(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    rmse = np.sqrt(np.mean((predictions - labels)**2))
    return {"rmse": rmse}

def get_metric(task_type, num_classes=None, ignore_index=None):
    if task_type == "classification":
        return compute_metrics_acc, "accuracy"
    elif task_type == "multilabel":
        return compute_metrics_mAP, "micro_mAP"
    elif task_type == "segmentation":
        if num_classes is None:
            raise ValueError("num_classes must be provided for segmentation task")
        if ignore_index is None:
            ignore_index = 255  # default ignore_index for segmentation
        if num_classes == 1:
            return compute_metrics_IoU_binary, "IoU"
        else:   
            return partial(compute_metrics_IoU, ignore_index=ignore_index, num_classes=num_classes), "IoU"
    elif task_type == "regression":
        return compute_metrics_rmse, "rmse"
    else:
        raise NotImplementedError
