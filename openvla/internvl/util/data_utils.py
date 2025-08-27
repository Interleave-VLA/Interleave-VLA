"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import loggamma

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int = 0
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        first = instances[0]

        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        if max_item_length > self.model_max_length:
            logging.warning("Some instances have longer length than `model_max_length`.")
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([self.pad_token_id] * max_item_length)
            temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            feat['attention_mask'] = feat['input_ids'].ne(self.pad_token_id)
            if 'labels' in feat:
                temp_labels[:feat['labels'].shape[0]] = feat['labels']
                feat['labels'] = temp_labels
            if 'position_ids' in feat:
                temp_position_ids = torch.LongTensor([self.pad_token_id] * max_item_length)
                temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids

        if 'labels' in first and first['labels'] is not None:
            if isinstance(first['labels'], torch.Tensor):
                batch['labels'] = torch.stack([f['labels'] for f in instances])
            else:
                dtype = torch.long if isinstance(first['labels'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['labels'] for f in instances], dtype=dtype)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all(
            [pv['pixel_values'] is not None for pv in instances]), "Invalid VLA Example with `pixel_values = None`!"

        if 'pixel_values' in first and first['pixel_values'] is not None:
            max_length = max(f['image_flags'].shape[0] for f in instances)
            for f in instances:
                f['image_flags'] = torch.cat([f['image_flags'], torch.zeros(max_length - f['image_flags'].shape[0], dtype=torch.bool)])
                f['pixel_values'] = torch.cat([f['pixel_values'],
                            torch.zeros(max_length - f['pixel_values'].shape[0], *f['pixel_values'].shape[1:])])

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('labels', 'dataset_name') and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])

        if "dataset_name" in first:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # Unfortunately, we cannot concat nor cuda `str` type
        # if dataset_names is not None:
        #     batch["dataset_names"] = dataset_names

        return batch
