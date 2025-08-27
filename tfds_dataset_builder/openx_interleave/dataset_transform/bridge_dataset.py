import tensorflow as tf

from .utils.language_decode import decode
from .utils.actions_transform import binarize_gripper_actions, relabel_actions

def dataset_transform(trajectory):
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

def get_action(x):
    return x['action']

def get_state(x):
    return x["observation"]["proprio"]

def get_observation(x):
    return x["observation"]["image_0"]

def get_prompt(x):
    return [decode(prompt) for prompt in x["language_instruction"]]
