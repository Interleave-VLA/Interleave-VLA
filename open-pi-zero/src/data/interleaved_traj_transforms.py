"""
Contains trajectory transforms used in the octo data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory
length).
"""

from typing import Optional

import tensorflow as tf

from .traj_transforms import (
    chunk_act_obs, subsample, pad_actions_and_proprio
)

def add_pad_mask_dict(traj: dict) -> dict:
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
                assert "language_instruction" in traj[key][subkey]
                # Try not to pad
                # pad_mask_dict[subkey] = { # 长度都是 traj_len，目的是之后通过 from_tensor_slices 分割
                #     "language_instruction": tf.strings.length(traj[key][subkey]["language_instruction"]) != 0,
                #     "image_instruction": tf.ones([traj_len], dtype=tf.bool),
                # }
            elif traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj