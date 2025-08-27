from typing import Iterator, Tuple, Any

import glob
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.nest as nest
import sys
import re
import random
from PIL import Image

from .conversion_utils import MultiThreadedDatasetBuilder
from .config import openx_config

tfds.core.utils.gcs_utils._is_gcs_disabled = True # Add this line to prevent `tfds build` from accessing google cloud storage

debug = True
# debug = False
random.seed(42)
IMAGE_PLACEHOLDER = "<image>"
sample_image_num = 1
dataset_name = "bc_z"
# bridge_dataset fractal20220817_data austin_sirius_dataset_converted_externally_to_rlds bc_z berkeley_autolab_ur5 iamlab_cmu_pickup_insert_converted_externally_to_rlds
# jaco_play language_table stanford_hydra_dataset_converted_externally_to_rlds ucsd_kitchen_dataset_converted_externally_to_rlds utaustin_mutex
# delted `droid``

RAW_DATA_PATH = openx_config[dataset_name]['path_to_dataset']
print(f"Loading from {RAW_DATA_PATH}")
dataset_transform = openx_config[dataset_name]['dataset_transform']
get_language_instruction = openx_config[dataset_name]['get_prompt']
get_observation = openx_config[dataset_name]['get_observation']
get_action = openx_config[dataset_name]['get_action']
get_state = openx_config[dataset_name]['get_state']

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def resize(img_array, target_size: tuple = (224, 224)) -> np.ndarray:
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray(img_array).convert('RGB')
        else:
            img = img_array
        img = img.resize(target_size) # different aspect ratio
        return np.array(img)
    
    def stack_tensors(*tensors):
        if not tensors:
            return None
        return tf.stack(tensors, axis=0)
    
    def _parse_example(episode_path):
        # load raw data
        with open(episode_path, 'rb') as f:
            data = pickle.load(f)

        _interleaved_prompt = {k.replace("_", " "): resize(img) 
                              for k, img in data['interleaved_prompt'].items() if img is not None}
        
        if "towel" in _interleaved_prompt:
            print(f"towel in {_interleaved_prompt.keys()}")
            return None
        
        episode = []
        trajectories = nest.map_structure(stack_tensors, *data['steps'])
        trajectories = dataset_transform(trajectories)
        states = get_state(trajectories)
        actions = get_action(trajectories)
        images = get_observation(trajectories)
        language_instructions = get_language_instruction(trajectories)
        assert len(states) == len(actions) == len(images) == len(language_instructions), "Length Mismatch!"
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            assert action.shape[-1] == 7, f"Action shape should be [x, y, z, r, p, y, gripper]"
            
            image = resize(images[i].numpy())
            
            language_instruction = language_instructions[i]
            original_instruction = language_instruction
            # print(f"prompt = {language_instruction}")
            
            interleaved_prompt = dict()
            for obj_name, img in _interleaved_prompt.items():
                if not obj_name.lower() in language_instruction.lower():
                    print(f"[ERROR] Object name not found in prompt. In {episode_path}, step {i}, language instruction: `{language_instruction}`, obj_name `{_interleaved_prompt.keys()}`")
                else:
                    interleaved_prompt[obj_name] = img
            
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
                    'image': image,
                    'state': np.asarray(state, dtype=np.float32),
                },
                'action': np.asarray(action, dtype=np.float32),
                'discount': 1,
                'reward': 0,
                'is_first': 0,
                'is_last': 0,
                'is_terminal': 0,
                'interleaved_instruction': {
                    'language_instruction': language_instruction,
                    'original_instruction': original_instruction,
                    'image_instruction': image_instruction,
                    'image_mask': image_mask
                }
            })
            # ======================= DEBUG =================================
            if debug:
                pass
                # from PIL import Image
                # print(episode[-1])
                # Image.fromarray(episode[-1]['observation']['image']).save("obs.jpg")
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
        if ret is not None:
            yield ret


class OpenXInterleaveDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
      '0.1.0': 'partial OpenX dataset.',
    }
    N_WORKERS = 1 if debug else 80 # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 320   # number of paths converted & stored in memory before writing to disk
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
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
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
        if debug:
            return {
                "train": glob.glob(RAW_DATA_PATH + "/train*.pkl")[: 1],
            }
        train_files = glob.glob(RAW_DATA_PATH + "/train*.pkl")
        val_files = glob.glob(RAW_DATA_PATH + "/val*.pkl") + glob.glob(RAW_DATA_PATH + "/test*.pkl")
        result = {"train": train_files}
        if val_files:
            result["val"] = val_files
        return result