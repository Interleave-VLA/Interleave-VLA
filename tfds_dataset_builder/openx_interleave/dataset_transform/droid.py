# FIXME

import tensorflow as tf
import numpy as np

def dataset_transform(trajectory):
    print(trajectory['observation'].keys())
    print("caresian position = ", trajectory['observation']['cartesian_position'])
    print("gripper position = ", trajectory['observation']['gripper_position'])
    print("joint position = ", trajectory['observation']['joint_position'])
    
    assert False
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["present/xyz"],
            trajectory["observation"]["present/axis_angle"],
            trajectory["observation"]["present/sensed_close"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory

def get_action(x):
    return x['action']

def get_state(x):
    return x["observation"]["proprio"]

def get_observation(x):
    return x["observation"]["image"]

def get_prompt(x):
    return [decode(prompt) for prompt in x["language_instruction"]]