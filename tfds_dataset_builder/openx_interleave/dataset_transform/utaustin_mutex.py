import tensorflow as tf
import numpy as np

from .utils.language_decode import decode
from .utils.actions_transform import invert_gripper_actions
from .utils.geometry_transform import quat2euler

def dataset_transform(trajectory):
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
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
