import tensorflow as tf
import numpy as np
from .utils.language_decode import decode
from .utils.actions_transform import rel2abs_gripper_actions
from .utils.geometry_transform import quat2euler

def dataset_transform(trajectory):
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
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

if __name__ == '__main__':
    import pickle
    import tensorflow.nest as nest
    
    file_path = "<path_to_fractal_dataset>/train00000_00.pkl"  # Replace with actual dataset path
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    traj = data['steps']
    def concat_or_stack_tensors(*tensors):
        if not tensors:
            return None        
        return tf.stack(tensors, axis=0)
    traj = nest.map_structure(concat_or_stack_tensors, *traj)
    # print(traj)
    traj = rt1_dataset_transform(traj)
    # print(traj)
    x = traj
    print(get_observation(x, 0))
    print(get_prompt(x, 0))
    print(get_state(x, 0))
    print(get_action(x, 0))