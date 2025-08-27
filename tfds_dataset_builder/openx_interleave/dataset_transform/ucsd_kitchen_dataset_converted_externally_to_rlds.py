import tensorflow as tf

from .utils.language_decode import decode

def dataset_transform(trajectory):
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :7]
    return trajectory

def get_action(x):
    return x['action']

def get_state(x):
    return x["observation"]["proprio"]

def get_observation(x):
    return x["observation"]["image"]

def get_prompt(x):
    return [decode(prompt) for prompt in x["language_instruction"]]
