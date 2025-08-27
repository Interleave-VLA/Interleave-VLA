import tensorflow as tf
import numpy as np
from .utils.language_decode import decode
from .utils.actions_transform import rel2abs_gripper_actions
from .utils.geometry_transform import quat2euler

def dataset_transform(trajectory):
    trajectory["observation"]["depth"] = trajectory["observation"].pop(
        "image_with_depth"
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"][
        :, 6:14
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory

def get_action(x):
    return x['action']

def get_state(x):
    states = []
    for state in x["observation"]["proprio"]:
        state = state.numpy()
        states.append(
            np.concatenate([state[: 3], quat2euler(state[3 : 7]), state[7 :]])
        )
    return states

def get_observation(x):
    return x["observation"]["image"]

def get_prompt(x):
    return [decode(prompt) for prompt in x["language_instruction"]]
