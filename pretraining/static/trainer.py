from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker
from functools import partial
import numpy as np
from typing import Dict
import torch
import datasets
from typing import Dict, Any, Optional, Union, Tuple, Callable
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from data_utils.temporal_sampler import BucketBatchSampler, AdaptiveBucketBatchSampler

class MAETrainer(Trainer):
    def __init__(self, modal_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.modal_mode = modal_mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.modal_mode == "random":
            modal = np.random.choice(['multi', 'optical', 'radar'])
        else:
            modal = self.modal_mode
            
        outputs = model(**inputs, modal = modal)
        
        assert self.compute_loss_func is not None, "compute_loss_func is not set"
        loss = self.compute_loss_func(outputs)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

class LEASTViTMAETrainer(MAETrainer):
    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        # MPS requrires forking if multiple workers are specified
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1

        dataloader_params = {
            # "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["batch_sampler"] = sampler_fn(dataset)
            # dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _has_length(self, dataset):
        """
        Checks if the dataset implements __len__() and it doesn't raise an error
        """
        try:
            return len(dataset) is not None
        except TypeError:
            # TypeError: len() of unsized object
            return False
        except AttributeError:
            # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
            return False

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not self._has_length(train_dataset):
            return None

        print(f"return AdaptiveBucketBatchSampler")
        assert train_dataset.frame_lengths is not None
        lengths = train_dataset.frame_lengths
        max_frames = 128
        boundaries = [1,2,4,6,8,12,16]
        return AdaptiveBucketBatchSampler(lengths=lengths, max_frames=max_frames, boundaries=boundaries)