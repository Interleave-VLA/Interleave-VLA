"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.traj_transforms import (
    chunk_act_obs,
    subsample
)

def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if subkey == "interleaved_instruction":
                # FIXME: Is this correct?
                """ Format
                'interleaved_instruction': tfds.features.FeaturesDict({
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction, with placeholders <image>.'
                    ),
                    'image_instruction': tfds.features.Sequence(
                        tfds.features.Image(
                            shape=(128, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Interleaved instruction images.'
                        ),
                        doc="Image sequence."
                    )
                })
                """
                assert "language_instruction" in traj[key][subkey]
                pad_mask_dict[subkey] = { # 长度都是 traj_len，目的是之后通过 from_tensor_slices 分割
                    "language_instruction": tf.strings.length(traj[key][subkey]["language_instruction"]) != 0,
                    "image_instruction": tf.ones([traj_len], dtype=tf.bool)
                }
            elif traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
