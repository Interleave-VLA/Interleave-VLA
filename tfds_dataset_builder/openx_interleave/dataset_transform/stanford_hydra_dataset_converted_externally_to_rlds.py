import tensorflow as tf

from .utils.language_decode import decode
from .utils.actions_transform import invert_gripper_actions

def dataset_transform(trajectory):
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            trajectory["observation"]["state"][:, -3:-2],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory

def get_action(x):
    return x['action']

def get_state(x):
    return x["observation"]["proprio"]

def get_observation(x):
    return x["observation"]["image"]

def get_prompt(x):
    return [decode(prompt) for prompt in x["language_instruction"]]
