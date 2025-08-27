from typing import Iterator, Tuple, Any

import glob
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import re
import random

from .conversion_utils import MultiThreadedDatasetBuilder
from .bridge_utils import resize

tfds.core.utils.gcs_utils._is_gcs_disabled = True # Add this line to prevent `tfds build` from accessing google cloud storage

random.seed(42)
IMAGE_PLACEHOLDER = "<image>"
sample_image_num = 1
RAW_DATA_PATH = "/path/to/generated/pickles" # See `interleave_pipeline/README.md` for details on how to generate interleaved pickle files

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path):
        # load raw data
        with open(episode_path, 'rb') as f:
            data = pickle.load(f)

        interleaved_prompt = {k.replace("_", " "): resize(img) 
                              for k, img in data['interleaved_prompt'].items() if img is not None}
        
        episode = []
        for i, step in enumerate(data['steps']):
            state = step['observation']['state']
            action = step['action']
            assert action.shape[-1] == 7, f"Action shape should be [x, y, z, r, p, y, gripper]"
            
            images = [resize(step['observation'][f'image_{i}'].numpy()) for i in range(2)]
            
            language_instruction = step['language_instruction']
            if not isinstance(language_instruction, str):
                language_instruction = language_instruction.numpy().decode('utf-8')
            original_instruction = language_instruction
            # print(f"prompt = {language_instruction}")
            assert all(obj_name.lower() in language_instruction.lower() for obj_name in interleaved_prompt.keys()), \
                f"Object name not found in prompt. In {episode_path}, step {i}, language instruction {language_instruction}, obj_name {interleaved_prompt.keys()}"
            
            sampled_keys = interleaved_prompt.keys()
            if len(interleaved_prompt) > sample_image_num:
                sampled_keys = random.sample(interleaved_prompt.keys(), sample_image_num)
            
            # sort in the order of appearance in the prompt
            obj_name_with_pos = []
            # Each on interleaved
            for obj_name in sampled_keys:
                match = re.search(re.escape(obj_name), language_instruction, flags=re.IGNORECASE)
                if match:
                    obj_name_with_pos.append((match.start(), obj_name))
                    language_instruction = re.sub(re.escape(obj_name), IMAGE_PLACEHOLDER, language_instruction, count=1, flags=re.IGNORECASE)
            # for obj_name in interleaved_prompt.keys():
            #     matches = re.finditer(re.escape(obj_name), language_instruction, flags=re.IGNORECASE)
            #     obj_name_with_pos.extend([(m.start(), obj_name) for m in matches])
            #     language_instruction = re.sub(re.escape(obj_name), IMAGE_PLACEHOLDER, language_instruction, flags=re.IGNORECASE)
            
            obj_name_with_pos = sorted(obj_name_with_pos, key=lambda x: x[0])
            image_instruction = [interleaved_prompt[obj_name] for _, obj_name in obj_name_with_pos]
            image_mask = [True] * len(image_instruction)
            if len(image_instruction) < sample_image_num:
                padding_image = np.zeros((224, 224, 3), dtype=np.uint8)
                padding_count = sample_image_num - len(image_instruction)
                image_instruction.extend([padding_image] * padding_count)
                image_mask.extend([False] * padding_count)
                assert (language_instruction.count(IMAGE_PLACEHOLDER) == sum(image_mask)), \
                        f"In {episode_path}: Unexpected number unmatch! {language_instruction}, {image_instruction}"
            else:
                assert (language_instruction.count(IMAGE_PLACEHOLDER) == len(image_instruction) == sample_image_num), \
                        f"In {episode_path}: Unexpected number unmatch! {language_instruction}, {image_instruction}"
            assert all(image.shape == (224, 224, 3) for image in image_instruction), f"Image shape is ..., expected (224, 224, 3). In {episode_path}, step {i}"
            # assert len(image_instruction) > 0, f"Expected at lease 1 image, got {len(image_instruction)}. In {episode_path}, step {i}"
            
            episode.append({
                'observation': {
                    'image_0': images[0], # consistent with configs.py in openvla
                    'image_1': images[1],
                    'state': np.asarray(state, dtype=np.float32),
                },
                'action': np.asarray(action, dtype=np.float32),
                'discount': step['discount'].numpy(),
                'reward': step['reward'].numpy(),
                'is_first': step['is_first'].numpy(),
                'is_last': step['is_last'].numpy(),
                'is_terminal': step['is_terminal'].numpy(),
                'interleaved_instruction': {
                    'language_instruction': language_instruction,
                    'original_instruction': original_instruction,
                    'image_instruction': image_instruction,
                    'image_mask': image_mask
                }
            })
            # ======================= DEBUG =================================
            # from PIL import Image
            # print(episode[-1])
            # Image.fromarray(episode[-1]['observation']['image_0']).save("obs.jpg")
            # for i, img in enumerate(episode[-1]['interleaved_instruction']['image_instruction']):
            #     Image.fromarray(img).save(f"{i}.jpg")
            # exit(0)
            
        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for path in paths:
        ret = _parse_example(path)
        yield ret


class BridgeInterleaveDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    # VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {
    #   '1.0.0': 'Initial release.',
    # }
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
      '0.1.0': 'partial bridge dataset.',
    }
    N_WORKERS = 60             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 160  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Secondary camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Current Robot EEF state (3D position, 4D quaternion, 1D gripper).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action (3D position, 4D quaternion, 1D gripper).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'interleaved_instruction': tfds.features.FeaturesDict({
                        'language_instruction': tfds.features.Text(
                            doc='Language Instruction, with placeholders <image>.'
                        ),
                        'original_instruction': tfds.features.Text(
                            doc='Language Instruction, without placeholders <image>.'
                        ),
                        'image_instruction': tfds.features.Sequence(
                            tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Interleaved instruction images.'
                            ),
                            # length=sample_image_num,
                            doc="Image sequence."
                        ),
                        'image_mask': tfds.features.Sequence(
                            tfds.features.Scalar(
                                dtype=np.bool_,
                                doc='Mask indicating whether the image is real (True) or padded (False)'
                            ),
                            # length=sample_image_num,
                            doc="Image mask sequence."
                        )
                    })
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            # "train": glob.glob(RAW_DATA_PATH + "/*.pkl"),
            "train": glob.glob(RAW_DATA_PATH + "/train*.pkl"),
            "val": glob.glob(RAW_DATA_PATH + "/val*.pkl"),
        }