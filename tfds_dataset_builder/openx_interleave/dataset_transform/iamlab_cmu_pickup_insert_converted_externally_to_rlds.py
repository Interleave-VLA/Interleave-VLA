import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tft
import numpy as np
from .utils.geometry_transform import quat2euler
from .utils.language_decode import decode

def dataset_transform(trajectory):
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
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
