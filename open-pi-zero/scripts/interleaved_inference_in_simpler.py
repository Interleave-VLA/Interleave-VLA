import os
import random
import time

import hydra
import imageio
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import torch
from omegaconf import OmegaConf

from src.model.vla.interleaved_pizero import InterleavedPiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time
from src.agent.env_adapter.interleaved_simpler import InterleavedBridgeSimplerAdapter

@log_execution_time()
def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def random_dance(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # simpler env
    env = simpler_env.make(args.task)

    # run an episode
    episode_id = random.randint(0, 20)
    env_reset_options = {}
    env_reset_options["obj_init_options"] = {
        "episode_id": episode_id,  # this determines the obj inits in bridge
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.get_language_instruction()
    
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # avoid tokenizer forking warning about deadlock
        )
        video_writer = imageio.get_writer(f"try_{args.task}_{episode_id}.mp4")
    print(
        f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
    )
    cnt_step = 0
    while 1:
        env_action = env.action_space.sample()
        obs, reward, success, truncated, info = env.step(env_action)
        cnt_step += 1
        if truncated:
            break

        # save frame
        img = get_image_from_maniskill2_obs_dict(env, obs)
        from PIL import Image
        Image.fromarray(img).save("obs.jpg")
        if args.recording:
            video_writer.append_data(img)

        # original octo eval only done when timeout, i.e., not upon success
        if truncated:
            if args.recording:
                video_writer.close()
            break

    # summary
    print(f"Task: {args.task}")
    print(f"Total environment steps: {cnt_step}")
    print(f"Success: {success}")
    if args.recording:
        print(f"Video saved as try_{args.task}_{episode_id}.mp4")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--recording", action="store_true")
    args = parser.parse_args()

    random_dance(args)
