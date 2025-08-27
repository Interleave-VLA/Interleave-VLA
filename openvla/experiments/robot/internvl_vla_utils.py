"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import torch

from internvl.extern.hf.conversation import Conversation as InternvlPromptBuilder, get_conv_template
from internvl.extern.hf.processing_internvl import InternvlProcessor
from internvl.extern.hf.configuration_internvl import OpenVLAConfig
from internvl.extern.hf.modeling_internvl import OpenVLAForActionPrediction
from internvl.vla.action_tokenizer import ActionTokenizer
from internvl.util.data_utils import PaddedCollatorForActionPrediction

from typing import Sequence, Dict

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PLACEHOLDER = "<image>"
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def get_vla_and_processor(model_name_or_path):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    config = OpenVLAConfig.from_pretrained(model_name_or_path)
    vla = OpenVLAForActionPrediction.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config,
        local_files_only=True,
    )
    vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(model_name_or_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    processor = InternvlProcessor(model_name_or_path, config)

    vla.img_context_token_id = processor.tokenizer.convert_tokens_to_ids(processor.IMG_CONTEXT_TOKEN)

    return vla, processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def prepare_vla_input(
    processor: InternvlProcessor,
    obs_img: np.ndarray,
    task_instruction: str, # interleaved instruction
    task_imgs: Sequence[np.ndarray],
    extra_obs: str = ""
):
    """Prepare input batch for the VLA policy."""
    concat_imgs = [obs_img] + task_imgs
    
    conversation = [
        {
            "from": "human",
            "value": f"Current observation is {IMAGE_PLACEHOLDER}{extra_obs}\nWhat action should the robot take to {task_instruction}?"
        },
        {
            "from": "gpt",
            "value": None
        }
    ]
    inputs = processor(images=concat_imgs, conversation=conversation)
    
    input_ids, pixel_values, attention_mask = inputs["input_ids"][0], inputs["pixel_values"], inputs["attention_mask"]

    image_end_token_id = processor.tokenizer.convert_tokens_to_ids(processor.IMG_END_TOKEN)[0]
    assert sum(input_ids == image_end_token_id) == len(
    concat_imgs), f"image tokens are truncated, this dataset is {dataset_name}."

    input_batch = dict(
        input_ids=input_ids,
        pixel_values=pixel_values.to(dtype=torch.bfloat16),
        image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
        attention_mask=attention_mask
    )
    return input_batch

def to_device(dct: Dict):
    for key, value in dct.items():
        dct[key] = value.to(DEVICE)
    return dct

def get_vla_action(
    vla: OpenVLAForActionPrediction,
    processor: InternvlProcessor,
    obs_img: np.ndarray,
    task_instruction: str, # interleaved instruction
    task_imgs: Sequence[np.ndarray],
    unnorm_key,
    extra_obs: str = ""
):
    """Generates an action with the VLA policy."""
    input_batch = prepare_vla_input(processor, obs_img, task_instruction, task_imgs, extra_obs=extra_obs)
    input_batch['input_ids'] = input_batch['input_ids'].unsqueeze(0)
    input_batch = to_device(input_batch)
    action = vla.predict_action(**input_batch, unnorm_key=unnorm_key, do_sample=False)
    return action.squeeze(0)

def get_batch_vla_action(
    vla: OpenVLAForActionPrediction,
    processor: InternvlProcessor,
    obs_imgs: Sequence[np.ndarray],
    task_instructions: Sequence[str], # interleaved instruction
    task_imgs: Sequence[Sequence[np.ndarray]],
    unnorm_key,
    extra_obss: Sequence[str] = ""
):
    if not extra_obss:
        extra_obss = [""] * len(obs_imgs)
    batches = [
        prepare_vla_input(processor, obs_img, task_instruction, task_img, extra_obs=extra_obs)
        for obs_img, task_instruction, task_img, extra_obs in zip(obs_imgs, task_instructions, task_imgs, extra_obss)
    ]
    input_batches = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length)(batches)
    input_batches['pixel_values'] = input_batches['pixel_values'].to(dtype=torch.bfloat16)
    input_batches = to_device(input_batches)
    actions = vla.predict_action(**input_batches, unnorm_key=unnorm_key, do_sample=False)
    return actions