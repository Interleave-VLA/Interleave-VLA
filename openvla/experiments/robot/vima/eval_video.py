"""
Video Evaluation Script for VIMA Robot Tasks

This script performs evaluation of VIMA models with video recording capabilities
for visual analysis and debugging of robot manipulation tasks.
"""

import sys
import os
import csv
import cv2
import time
from pyvirtualdisplay.smartdisplay import SmartDisplay
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import logging
from typing import Dict, Tuple

# Add project paths to system path
# Note: Update these paths according to your project structure
sys.path.append('path/to/openvla')
sys.path.append('path/to/vima_env')

from experiments.robot.internvl_vla_utils import get_vla_and_processor, get_vla_action
from vima_utils import (
    create_env, prepare_obs, prepare_prompt_images, prepare_prompt, clip_action_se2,
    transform_vla_to_quat, transform_quat_to_vima, transform_vima_to_quat, 
    clip_action, qmul
)

# Configuration constants
DEFAULT_CSV_PATH = 'results/video_evaluation_results.csv'
DEFAULT_DATASET_NAME = "vima_se2_dataset"
DEFAULT_MAX_STEPS = 3
DEFAULT_NUM_EVAL = 1
DEFAULT_ENV_SEED = 42
DEFAULT_TARGET_SIZE = (224, 224)
DEFAULT_RECORD = True

# Task definitions
TASK_LIST = {
    1: 'visual_manipulation',
    2: 'scene_understanding',
    3: 'rotate',
    4: 'rearrange',
    7: 'novel_noun',
    10: 'follow_motion',
    11: 'follow_order',
    14: 'same_texture',
    15: 'same_shape',
}

# Partition definitions
PARTITION_LIST = ['placement_generalization', 'combinatorial_generalization']

# Model configurations
DEFAULT_CKPT_LIST = [4400]
DEFAULT_TASK_IDS = [2]


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_video_writer(video_path: str, target_size: Tuple[int, int], fps: float = 30.0):
    """
    Create OpenCV video writer for recording.
    
    Args:
        video_path: Path to save the video
        target_size: Target video dimensions (width, height)
        fps: Frames per second
        
    Returns:
        OpenCV VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        filename=video_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(target_size[1], target_size[0])  # OpenCV uses (width, height)
    )
    return video_writer


def process_action_bounds(meta_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process action bounds from environment metadata.
    
    Args:
        meta_info: Environment metadata containing action bounds
        
    Returns:
        Tuple of (low_bounds, high_bounds) tensors
    """
    action_bounds = [meta_info["action_bounds"]]
    action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
    action_bounds_high = [action_bound["high"] for action_bound in action_bounds]
    
    action_bounds_low = torch.tensor(action_bounds_low, dtype=torch.float32)
    action_bounds_high = torch.tensor(action_bounds_high, dtype=torch.float32)
    
    return action_bounds_low, action_bounds_high


def create_action_dict(action: np.ndarray, action_bounds_low: torch.Tensor, 
                      action_bounds_high: torch.Tensor) -> Dict:
    """
    Create action dictionary from raw action array.
    
    Args:
        action: Raw action array
        action_bounds_low: Lower bounds for actions
        action_bounds_high: Upper bounds for actions
        
    Returns:
        Action dictionary with position and rotation components
    """
    actions = {}
    
    # Position actions
    actions["pose0_position"] = torch.tensor(action[:2], dtype=torch.float32)
    actions["pose1_position"] = torch.tensor(action[6:8], dtype=torch.float32)
    
    # Clamp position actions to bounds
    actions["pose0_position"] = torch.clamp(
        actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
    ).squeeze(0)
    actions["pose1_position"] = torch.clamp(
        actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
    ).squeeze(0)
    
    # Rotation actions
    actions["pose0_rotation"] = torch.tensor(action[2:6], dtype=torch.float32)
    actions["pose1_rotation"] = torch.tensor(action[8:], dtype=torch.float32)
    
    # Clamp rotation actions to [-1, 1]
    actions["pose0_rotation"] = torch.clamp(
        actions["pose0_rotation"], min=-1, max=1
    ).squeeze(0)
    actions["pose1_rotation"] = torch.clamp(
        actions["pose1_rotation"], min=-1, max=1
    ).squeeze(0)
    
    # Convert to numpy
    actions = {k: v.cpu().numpy() for k, v in actions.items()}
    
    return actions


def save_results_to_csv(csv_path: str, row_data: Dict):
    """
    Save evaluation results to CSV file.
    
    Args:
        csv_path: Path to CSV file
        row_data: Dictionary containing result data
    """
    file_exists = os.path.exists(csv_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def eval_one_model(ckpt: int, model: str, task: str, partition: str, 
                  dataset_name: str, csv_path: str):
    """
    Evaluate a single model checkpoint with video recording.
    
    Args:
        ckpt: Checkpoint number
        model: Model name
        task: Task name
        partition: Dataset partition
        dataset_name: Dataset name for normalization
        csv_path: Path to save CSV results
    """
    model_name_or_path = f"path/to/model/runs/{model}/{ckpt}_chkpt"
    
    # Setup video recording
    video_writer = None
    if DEFAULT_RECORD:
        video_path = f"videos/{model}_{ckpt}_{task}_{partition}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = create_video_writer(video_path, DEFAULT_TARGET_SIZE)

    try:
        # Load model
        vla, processor = get_vla_and_processor(model_name_or_path)

        # Create environment
        env = create_env(task, partition, seed=DEFAULT_ENV_SEED)
        success = 0

        # Run evaluation
        with SmartDisplay(visible=False, size=(1024, 768)) as disp:
            for _ in tqdm(range(DEFAULT_NUM_EVAL), desc=f"Evaluating {model}_{ckpt}"):
                obs = env.reset()
                
                meta_info = env.meta_info
                prompt = env.prompt
                prompt_assets = prepare_prompt_images(env.prompt_assets, image_size=DEFAULT_TARGET_SIZE)
                prompt_img = env.get_multi_modal_prompt_img()

                task_instruction, task_imgs = prepare_prompt(
                    prompt, prompt_assets, target_size=DEFAULT_TARGET_SIZE
                )
                
                # Main evaluation loop
                for elapsed_steps in tqdm(range(DEFAULT_MAX_STEPS), desc="Steps", leave=False):
                    obs_img = prepare_obs(obs, target_size=DEFAULT_TARGET_SIZE)
                    
                    # Record video frame
                    if DEFAULT_RECORD and video_writer:
                        frame = cv2.cvtColor(np.array(obs_img), cv2.COLOR_RGB2BGR)
                        video_writer.write(frame)
                        obs_img.save(f"frames/{model}_{ckpt}_obs.jpg")
                    
                    # Get action from model
                    action = get_vla_action(
                        vla, processor, obs_img, task_instruction, task_imgs, 
                        unnorm_key=dataset_name
                    )
                    
                    # Process action bounds and create action dict
                    action_bounds_low, action_bounds_high = process_action_bounds(meta_info)
                    actions = create_action_dict(action, action_bounds_low, action_bounds_high)
                    
                    # Execute action
                    obs, _, done, info = env.step(actions)
                    logging.info(f"Step {elapsed_steps}, action = {actions}")
                    
                    if info['success']:
                        break
                        
                success += info['success']
                logging.info(f"Success: {success}")

        # Clean up
        env.close()
        
        # Calculate success rate
        success_rate = success / DEFAULT_NUM_EVAL
        logging.info(f"Success Rate: {success_rate}, success: {success}, total: {DEFAULT_NUM_EVAL}")
        
        # Save results to CSV
        model_dir = model_name_or_path.split('/')[-2]
        model_name = model_name_or_path.split('/')[-1]

        row_data = {
            "model_dir": model_dir,
            "model_name": model_name,
            "success_rate": success_rate,
            "task": task,
            "partition": partition,
            "success": success,
            "total": DEFAULT_NUM_EVAL
        }

        save_results_to_csv(csv_path, row_data)
        
    except Exception as e:
        logging.error(f"Error evaluating {model}_{ckpt}: {e}")
    finally:
        # Clean up video resources
        if video_writer:
            video_writer.release()


def main():
    """Main function to run video evaluation."""
    setup_logging()
    
    # Create results directory
    os.makedirs(os.path.dirname(DEFAULT_CSV_PATH), exist_ok=True)
    
    # Run evaluation for all models and tasks
    for task_id in DEFAULT_TASK_IDS:
        model = f'0327-task{task_id}-se2'
        dataset_name = f'se2_task{task_id}'
        
        for ckpt in DEFAULT_CKPT_LIST:
            for partition in PARTITION_LIST:
                eval_one_model(
                    ckpt, model, TASK_LIST[task_id], partition, 
                    dataset_name, DEFAULT_CSV_PATH
                )


if __name__ == '__main__':
    main()