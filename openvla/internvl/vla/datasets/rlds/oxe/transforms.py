"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any, Dict

import tensorflow as tf
from prismatic.vla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)
from internvl.vla.datasets.rlds.oxe.utils.vima_utils import quaternion_to_rpy

def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation" or key == "interleaved_instruction":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory

def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory

def vima_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Note that we have calculated delta action according to EE_POSE controller
    # quaternion to rpy euler angles
    trajectory["action"] = tf.concat([
        trajectory["action"][:, : 3],
        quaternion_to_rpy(trajectory["action"][:, 3 : 7]),
        trajectory["action"][:, 7 :]
    ], axis=1)
    trajectory["observation"]["state"] = tf.concat([
        trajectory["observation"]["state"][:, : 3],
        quaternion_to_rpy(trajectory["observation"]["state"][:, 3 : 7]),
        trajectory["observation"]["state"][:, 7 :]
    ], axis=1)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, : 6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1 :]
    return trajectory

def vima_se2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory

def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    ### OXE
    "stanford_hydra_dataset_converted_externally_to_rlds:0.1.0": stanford_hydra_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    ### vima datasets (modified versions)
    "vima_dataset": vima_dataset_transform,
    "vima_dataset:0.1.0": vima_dataset_transform,
    "vima_task2_dataset": vima_dataset_transform,
    "vima_task2_dataset:0.1.0": vima_dataset_transform,
    "vima_task7_dataset": vima_dataset_transform,
    "vima_task7_dataset:0.1.0": vima_dataset_transform,
    "vima_se2_dataset": vima_se2_dataset_transform,
    "vima_se2_dataset:0.1.0": vima_se2_dataset_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
}
