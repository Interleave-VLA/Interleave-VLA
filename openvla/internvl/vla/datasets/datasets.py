"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch
from PIL import Image
from pandas import concat
from torch.utils.data import Dataset, IterableDataset
import tensorflow as tf

from internvl.extern.hf.conversation import Conversation
from internvl.extern.hf.processing_internvl import InternvlProcessor
from prismatic.util.data_utils import tree_map
from internvl.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from internvl.vla.datasets.rlds.oxe import INTERLEAVED_MIXTURES, OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from internvl.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset

IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "<image>"


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    internvl_processor: InternvlProcessor

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, cur_state, action = rlds_batch["dataset_name"], rlds_batch["state"], rlds_batch["action"][0]
        dataset_name, cur_state, action = rlds_batch["dataset_name"], rlds_batch.get("state", None), rlds_batch["action"][0]
        obs_img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        
        if "interleaved_instruction" in rlds_batch["task"]:
            task_lang = rlds_batch["task"]["interleaved_instruction"]["language_instruction"].decode().lower()
            assert IMAGE_PLACEHOLDER in task_lang, f"Expected '{IMAGE_PLACEHOLDER}' in task language!"
            task_imgs = [Image.fromarray(img) for img in rlds_batch["task"]["interleaved_instruction"]["image_instruction"]]
            assert task_lang.count(IMAGE_PLACEHOLDER) == len(
                task_imgs), "Number of images doesn't match in task instruction!"
            concat_imgs = [obs_img] + task_imgs
        else:
            task_lang = rlds_batch["task"]["language_instruction"].decode().lower()
            concat_imgs = [obs_img]

        # TODO: Do we want to support video instructions?
        extra_obs = f", suction status is {int(cur_state[-1])}" if "vima_dataset" in dataset_name.decode('utf-8') else ""
        conversation = [
            {"from": "human",
             "value": f"Current observation is {IMAGE_PLACEHOLDER}{extra_obs}\nWhat action should the robot take to {task_lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        # you can check if the action tokenizer is correct by viewing
        # self.action_tokenizer.decode_token_ids_to_actions(np.array(self.action_tokenizer.tokenizer(self.action_tokenizer(action))['input_ids']))
        inputs = self.internvl_processor(images=concat_imgs, conversation=conversation)

        input_ids, labels, attention_mask, pixel_values = inputs["input_ids"][0], inputs["labels"][0], \
            inputs["attention_mask"][0], inputs["pixel_values"]

        # Calculate position_ids for packed dataset
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        image_end_token_id = \
            self.internvl_processor.tokenizer.convert_tokens_to_ids(self.internvl_processor.IMG_END_TOKEN)[0]
        assert sum(input_ids == image_end_token_id) == len(
            concat_imgs), f"image tokens are truncated, this dataset is {dataset_name}."
        # Return tensors' dimension should be 1
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
            dataset_name=dataset_name
        )


class RLDSDataset(IterableDataset):
    def __init__(
            self,
            data_root_dir: Path,
            data_mix: str,
            batch_transform: RLDSBatchTransform,
            shuffle_buffer_size: int = 256_000,
            train: bool = True,
            image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        interleaved_instruction = False
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        elif data_mix.startswith('se2'):
            mixture_spec = INTERLEAVED_MIXTURES['se2_dataset']
            mixture_spec = [(data_mix, 1.0)]
            interleaved_instruction = True
        elif self.data_mix in INTERLEAVED_MIXTURES:
            mixture_spec = INTERLEAVED_MIXTURES[self.data_mix]
            interleaved_instruction = True
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
        logging.info(f"data mix {self.data_mix} interleaved instruction status: {interleaved_instruction}")

        # fmt: off
        # 根据 dataset 的 config 获取 meta 信息
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language_instruction=(not interleaved_instruction),
            load_interleaved_instruction=interleaved_instruction,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        print('###########################\ndataset_kwargs',per_dataset_kwargs)
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # If we wanted to feed / predict more than one step
                future_action_window_size=0,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=None,  # For Internvl Models, we DO NOT need to resize images to default size
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=False,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": dict(
                # random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                # random_brightness=[0.2],
                # random_contrast=[0.8, 1.2],
                # random_saturation=[0.8, 1.2],
                # random_hue=[0.05],
                
                # random_brightness=[0.5],          # 亮度 ±50%
                # random_contrast=[0.5, 1.5],        # 对比度 0.5x~1.5x
                # random_saturation=[0.5, 1.5],      # 饱和度 0.5x~1.5x
                # random_hue=[0.15],
                
                random_brightness=[0.8],          # 亮度 ±50%
                random_contrast=[0.2, 1.8],        # 对比度 0.5x~1.5x
                random_saturation=[0.2, 1.8],      # 饱和度 0.5x~1.5x
                random_hue=[0.3],
                augment_order=[
                    # "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        for rlds_batch in dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out
