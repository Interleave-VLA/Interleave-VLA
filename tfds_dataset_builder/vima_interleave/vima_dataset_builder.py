from typing import Iterator, Tuple, Any

import glob
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import sys

from .conversion_utils import MultiThreadedDatasetBuilder
from .vima_utils import qmul, qdiv, resize

tfds.core.utils.gcs_utils._is_gcs_disabled = True # Add this line to prevent `tfds build` from accessing google cloud storage

IMAGE_PLACEHOLDER = "<image>"

RAW_DATA_PATH = "/path/to/collected/vima/dataset/*/"

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    # print("#########Generating examples from paths:", paths)
    def _parse_example(episode_path):
        # load raw data
        data = np.load(episode_path, allow_pickle=True)

        episode = []
        for i, step in enumerate(data):
            state2 = step['action']
            # state1 = step['observation']['state'] 
            assert state2.shape[-1] == 12, f"Action shape should be [x, y, 4-rotate] * 2" #
            # action = qdiv(state2, state1)
            # action = np.concatenate([action, [state2[7]]], axis=-1)
            action = state2
            
            image = step['observation']['image']
            assert image.shape == (128, 256, 3), f"Image shape is {image.shape}, expected (128, 256, 3). In {episode_path}, step {i}"
            image = resize(image)
            
            language_instruction = step['language_instruction']
            
            image_instruction = step['image_instruction']
            assert all(image.shape == (224, 224, 3) for image in image_instruction), f"Image shape is {image.shape}, expected (128, 256, 3). In {episode_path}, step {i}"
            # image_instruction = [resize(image) for image in image_instruction]
            
            assert language_instruction.count(IMAGE_PLACEHOLDER) == len(image_instruction), f"Mismatch between language and image instruction lengths. \
                    In {episode_path}, prompt: {language_instruction}"
            episode.append({
                'observation': {
                    'image': image,
                    # 'state': np.asarray(state1, dtype=np.float32),
                },
                'action': np.asarray(action, dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == len(data) - 1),
                'is_first': i == 0,
                'is_last': i == (len(data) - 1),
                'is_terminal': i == (len(data) - 1),
                'interleaved_instruction': {
                    'language_instruction': language_instruction,
                    'image_instruction': image_instruction
                }
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }
        print()

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for path in paths:
        ret = _parse_example(path)
        yield ret


class VIMADataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    # VERSION = tfds.core.Version('1.0.0')
    # RELEASE_NOTES = {
    #   '1.0.0': 'Initial release.',
    # }
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
      '0.1.0': 'partial vima dataset.',
    }
    N_WORKERS = 2            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3), # (128, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(12,),
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
                        'image_instruction': tfds.features.Sequence(
                            tfds.features.Image(
                                shape=(224, 224, 3), # (128, 256, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Interleaved instruction images.'
                            ),
                            doc="Image sequence."
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
            "train": glob.glob(RAW_DATA_PATH + "/*.npy")
        }