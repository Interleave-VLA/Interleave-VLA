# Copied the functions from https://huggingface.co/OpenGVLab/InternVL2_5-8B#inference-with-transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
# from transformers.models.fuyu.convert_fuyu_model_weights_to_hf import tokenizer_class
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType

from typing import List, Optional, Union, Sequence, Dict

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from internvl.extern.hf.tokenization_internlm2 import InternLM2Tokenizer
from internvl.extern.hf.tokenization_internlm2_fast import InternLM2TokenizerFast

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    # For image of 256*128, it turns out to be 3 images
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_image_from_file(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    return process_image(image, input_size=input_size, max_num=max_num)


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternvlProcessor:
    r"""
    Constructs a Internvl processor which wraps a Internvl image processor and a Internvl tokenizer into a single
    processor.
    Borrows idea from https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset.py#L711
    """
    IMG_START_TOKEN = '<img>',
    IMG_END_TOKEN = '</img>',
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IGNORE_TOKEN_ID = -100
    tokenizer_class = InternLM2Tokenizer

    def __init__(self, pretrained_model_name_or_path, config):
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_model_name_or_path, use_fast=False)
        self.config = config

    def __call__(
            self,
            conversation: Sequence[Dict] = None,
            images: ImageInput = None,
            max_length: int = None,
            force_image_size: int = 224
    ) -> BatchFeature:
        assert isinstance(conversation, list) and isinstance(conversation[0],
                                                             dict), "conversation must be a list of dictionaries!"

        if conversation[0]['from'] == 'system':
            system_prompt = conversation[0]['value']
            conversation = conversation[1:]  # remove system prompt
        else:
            system_prompt = None

        data = dict()

        # Process images
        if images is not None:
            if isinstance(images[0], str):
                pixel_values = [load_image_from_file(image) for image in images]
            else:
                # force image size to be 224x224
                assert all(x.size[0] == force_image_size and x.size[1] == force_image_size for x in images)
                pixel_values = [process_image(image, input_size=force_image_size) for image in images]
                # # resize to 224x224, this is subject to change!
                # pixel_values = [process_image(image, input_size=min(force_image_size, max(image.size))) for image in images]
            num_patches_list = [image.size(0) for image in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            iter_pv = iter(pixel_values)
            data["pixel_values"] = pixel_values
            data["num_patches_list"] = num_patches_list
            # Process Text
            new_conversation = []
            current_image_idx = 0
            for conv in conversation:
                if conv['from'] == 'human':
                    image_cnt = conv['value'].count('<image>')
                    for i in range(image_cnt):
                        if current_image_idx == len(num_patches_list):
                            break
                        _, w, h = next(iter_pv).shape
                        ww, hh = w // self.config.vision_config.patch_size, h // self.config.vision_config.patch_size # conv layer
                        num_image_token = int(ww * self.config.downsample_ratio) * int(hh * self.config.downsample_ratio)
                        image_tokens = f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_patches_list[current_image_idx] * num_image_token}{self.IMG_END_TOKEN}'
                        conv['value'] = conv['value'].replace('<image>', image_tokens, 1)
                        current_image_idx += 1
                new_conversation.append(conv)
            conversation = new_conversation
            assert current_image_idx == len(num_patches_list), f'{current_image_idx} != {len(num_patches_list)}'
        else:
            pixel_values = None

        # Convert conversation to prompt
        batches, roles = [], []
        if system_prompt is not None:
            batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
            roles.append('system')
        for conv in conversation:
            if conv['from'] == 'human':
                batches.append(f'<|im_start|>user\n{conv["value"]}<|im_end|>\n')
                roles.append('human')
            elif conv['from'] == 'gpt':
                if conv["value"]: # train
                    batches.append(f'<|im_start|>assistant\n{conv["value"]}<|im_end|>\n')
                else: # eval
                    batches.append(f'<|im_start|>assistant\n')
                roles.append('gpt')
            else:
                raise NotImplementedError

        add_bos_token = getattr(self.tokenizer, 'add_bos_token', False)
        if add_bos_token:  # for InternLM series
            batches[0] = self.tokenizer.bos_token + batches[0]

        # Tokenize conversations
        input_ids = self.tokenizer(
            batches,
            return_tensors='np',
            padding=False,
            max_length=self.tokenizer.model_max_length,
            truncation=False,
        ).input_ids

        if add_bos_token:  # for InternLM series
            input_ids = [item[1:] for item in input_ids]

        final_input_ids, final_targets = [], []
        ignore_ids = self.tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
        ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role == 'system' or role == 'human':
                final_targets.append(torch.full(input_id.shape, self.IGNORE_TOKEN_ID))  # ignore
            elif role == 'gpt':
                target = input_id.copy()
                target[:ignore_len] = self.IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
                # by default, no need to predict <|im_end|>
                target[-2:] = self.IGNORE_TOKEN_ID  # ignore loss for `<|im_end|>\n`
                final_targets.append(target)
            else:
                raise NotImplementedError

        input_ids = torch.tensor(np.concatenate(final_input_ids))[:self.tokenizer.model_max_length]
        targets = torch.tensor(np.concatenate(final_targets))[:self.tokenizer.model_max_length]
        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)

        data = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
        )
        return BatchFeature(data=data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
