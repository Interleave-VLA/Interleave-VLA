"""
VIMA (Visual Manipulation) Utilities Module

This module provides utility functions for VIMA robot manipulation tasks including:
- Image preprocessing and prompt preparation
- Action space transformations between different formats
- Environment creation and management
- Video recording capabilities
"""

from typing import Sequence, Dict, Tuple, Optional, Union
import os
from copy import deepcopy

from einops import rearrange
import numpy as np
from scipy.spatial.transform import Rotation as R

from PIL import Image
import cv2

from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
from vima_bench import make, PARTITION_TO_SPECS

# Constants
IMAGE_PLACEHOLDER = "<image>"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MAX_RETRIES = 10


def prepare_obs(obs: Dict, view: str = 'front', target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Prepare observation image from environment observation.
    
    Args:
        obs: Environment observation dictionary
        view: Camera view to use ('front', 'side', etc.)
        target_size: Target size for image resizing (width, height)
        
    Returns:
        PIL Image object
    """
    obs_img = rearrange(obs['rgb'][view], "c h w -> h w c")
    obs_img = Image.fromarray(obs_img)
    if target_size is not None:
        obs_img = obs_img.resize(target_size)
    return obs_img


def prepare_prompt_images(
    prompt_assets: Dict, 
    view: str = 'front', 
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Dict:
    """
    Prepare prompt images by cropping and resizing based on segmentation.
    
    Args:
        prompt_assets: Dictionary containing prompt assets with RGB and segmentation data
        view: Camera view to use
        image_size: Target size for cropped images (width, height)
        
    Returns:
        Dictionary with processed prompt images
    """
    prompt_imgs = deepcopy(prompt_assets)
    
    for asset_name, asset in prompt_assets.items():
        obj_info = asset["segm"]["obj_info"]
        placeholder_type = asset["placeholder_type"]
        rgb_this_view = asset["rgb"][view]
        segm_this_view = asset["segm"][view]
        
        if placeholder_type == "object":
            obj_id = obj_info["obj_id"]
            ys, xs = np.nonzero(segm_this_view == obj_id)
            
            if len(xs) < 2 or len(ys) < 2:
                continue
                
            xmin, xmax = np.min(xs), np.max(xs)
            ymin, ymax = np.min(ys), np.max(ys)
            cropped_img = rgb_this_view[:, ymin:ymax + 1, xmin:xmax + 1]
            
            # Make image square by padding if necessary
            if cropped_img.shape[1] != cropped_img.shape[2]:
                diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                
                if cropped_img.shape[1] > cropped_img.shape[2]:
                    pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                else:
                    pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    
                cropped_img = np.pad(
                    cropped_img,
                    pad_width,
                    mode="constant",
                    constant_values=255,
                )
                assert cropped_img.shape[1] == cropped_img.shape[2], "Cropped image should be square"
                
        elif placeholder_type == "scene":
            cropped_img = rgb_this_view
        else:
            raise ValueError(f"Unknown placeholder type: {placeholder_type}")
            
        # Resize to target size
        cropped_img = rearrange(cropped_img, "c h w -> h w c")
        cropped_img = np.asarray(cropped_img)
        cropped_img = cv2.resize(
            cropped_img,
            image_size,
            interpolation=cv2.INTER_AREA,
        )
        cropped_img = rearrange(cropped_img, "h w c -> c h w")
        prompt_imgs[asset_name]["rgb"][view] = cropped_img
        
    return prompt_imgs


def prepare_prompt(
    prompt: str, 
    prompt_assets: Dict, 
    view: str = 'front', 
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[str, list]:
    """
    Prepare prompt text and associated images.
    
    Args:
        prompt: Text prompt with placeholders
        prompt_assets: Dictionary containing prompt assets
        view: Camera view to use
        target_size: Target size for images
        
    Returns:
        Tuple of (processed_prompt, prompt_images)
    """
    prompt_img_with_pos = []
    
    for placeholder in prompt_assets.keys():
        # Find all positions of this placeholder in the prompt
        positions = [pos for pos in range(len(prompt)) if prompt.startswith(placeholder, pos)]
        prompt_img_with_pos.extend([(pos, placeholder) for pos in positions])
        
        # Replace placeholder with image token
        placeholder_text = '{' + placeholder + '}'
        prompt = prompt.replace(placeholder_text, IMAGE_PLACEHOLDER)
    
    # Sort by position and prepare images
    prompt_img_with_pos = sorted(prompt_img_with_pos, key=lambda x: x[0])
    prompt_imgs = [
        rearrange(prompt_assets[pair[1]]['rgb'][view], "c h w -> h w c") 
        for pair in prompt_img_with_pos
    ]
    prompt_imgs = [Image.fromarray(img) for img in prompt_imgs]
    
    if target_size is not None:
        prompt_imgs = [img.resize(target_size) for img in prompt_imgs]
        
    assert len(prompt_imgs) == prompt.count(IMAGE_PLACEHOLDER), \
        f"Mismatch between language and image instruction lengths. prompt: {prompt}."
        
    return prompt, prompt_imgs


def transform_vla_to_quat(action: np.ndarray) -> np.ndarray:
    """
    Transform VLA action format to quaternion format.
    
    Args:
        action: Action array in VLA format [x, y, z, rx, ry, rz, gripper]
        
    Returns:
        Action array in quaternion format [x, y, z, qw, qx, qy, qz, gripper]
    """
    quat_action = np.concatenate([
        action[:3].astype(np.float32),
        R.from_euler('xyz', action[3:6], degrees=False).as_quat().astype(np.float32),
        [np.round(action[6]).astype(int)]
    ])
    return quat_action


def transform_quat_to_vima(action: np.ndarray) -> Dict:
    """
    Transform quaternion action format to VIMA format.
    
    Args:
        action: Action array in quaternion format [x, y, z, qw, qx, qy, qz, gripper]
        
    Returns:
        Action dictionary with position, rotation, and gripper
    """
    vima_action = {
        "position": action[:3].astype(np.float32),
        "rotation": action[3:7].astype(np.float32),
        "gripper": np.round(action[7]).astype(int)
    }
    return vima_action


def transform_vima_to_quat(action: Dict) -> np.ndarray:
    """
    Transform VIMA action format to quaternion format.
    
    Args:
        action: Action dictionary with position, rotation, and gripper
        
    Returns:
        Action array in quaternion format
    """
    position = np.array(action["position"], dtype=np.float32)
    rotation = np.array(action["rotation"], dtype=np.float32)
    gripper = np.array(action.get("gripper", 0), dtype=np.float32)
    quat_action = np.concatenate([position, rotation, [gripper]])
    return quat_action


def transform_vla_to_vima(action: np.ndarray) -> Dict:
    """
    Transform VLA action format to VIMA format.
    
    Args:
        action: Action array in VLA format [x, y, z, rx, ry, rz, gripper]
        
    Returns:
        Action dictionary in VIMA format
    """
    vima_action = {
        "position": action[:3].astype(np.float32),
        "rotation": R.from_euler('xyz', action[3:6], degrees=False).as_quat().astype(np.float32),
        "gripper": np.round(action[6]).astype(int)
    }
    return vima_action


def transform_vima_to_vla(action: Dict) -> np.ndarray:
    """
    Transform VIMA action format to VLA format.
    
    Args:
        action: Action dictionary with position, rotation, and gripper
        
    Returns:
        Action array in VLA format [x, y, z, rx, ry, rz, gripper]
    """
    position = np.array(action["position"], dtype=np.float32)
    rotation = R.from_quat(action["rotation"]).as_euler('xyz', degrees=False).astype(np.float32)
    gripper = np.array(action.get("gripper", 0), dtype=np.float32)
    vla_action = np.concatenate([position, rotation, [gripper]])
    return vla_action


def clip_action(action: Dict, action_space: Dict) -> Dict:
    """
    Clip action values to action space bounds.
    
    Args:
        action: Action dictionary
        action_space: Action space definition with low/high bounds
        
    Returns:
        Clipped action dictionary
    """
    return {
        k: np.clip(v, action_space[k].low, action_space[k].high) 
        if k in ["position", "rotation"] else v 
        for k, v in action.items()
    }


def clip_action_se2(action: Dict, action_space: Dict) -> Dict:
    """
    Clip SE2 action values to action space bounds.
    
    Args:
        action: SE2 action dictionary
        action_space: Action space definition
        
    Returns:
        Clipped SE2 action dictionary
    """
    action_space['position'] = action_space['pose0_position']
    action_space['rotation'] = action_space['pose0_rotation']
    return {
        k: np.clip(v, action_space[k].low, action_space[k].high) 
        if k in ["position", "rotation"] else v 
        for k, v in action.items()
    }


def qmul(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Multiply two poses (position + quaternion).
    
    Args:
        pose1: First pose [x, y, z, qw, qx, qy, qz]
        pose2: Second pose [x, y, z, qw, qx, qy, qz]
        
    Returns:
        Combined pose [x, y, z, qw, qx, qy, qz]
    """
    p1, q1 = pose1[..., :3], R.from_quat(pose1[..., 3:7])
    p2, q2 = pose2[..., :3], R.from_quat(pose2[..., 3:7])
    
    new_q = q1 * q2  # quaternion multiplication
    new_p = p1 + q1.apply(p2)  # position transformation
    
    return np.concatenate([new_p, new_q.as_quat()], axis=-1)


class ResetFaultToleranceWrapper(Wrapper):
    """
    Wrapper that provides fault tolerance for environment resets.
    Retries reset operations multiple times before giving up.
    """
    
    def __init__(self, env, max_retries: int = DEFAULT_MAX_RETRIES):
        super().__init__(env)
        self.max_retries = max_retries

    def reset(self):
        """Reset environment with fault tolerance."""
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except Exception:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
                
        raise RuntimeError(
            f"Failed to reset environment after {self.max_retries} retries"
        )


class TimeLimitWrapper(_TimeLimit):
    """
    Wrapper that adds time limits to environments.
    Extends the default time limit with bonus steps.
    """
    
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


def create_env(
    task: str, 
    partition: str, 
    record_gui: bool = False, 
    seed: int = 42
):
    """
    Create a VIMA environment with appropriate wrappers.
    
    Args:
        task: Task name
        partition: Dataset partition
        record_gui: Whether to record GUI
        seed: Random seed
        
    Returns:
        Wrapped environment
    """
    return TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][partition][task],
                seed=seed,
                render_prompt=True,
                display_debug_window=False,
                hide_arm_rgb=False,
            )
        ),
        bonus_steps=2,
    )


class VideoRecorder:
    """
    Context manager for recording video from PIL images.
    Automatically handles video encoding and cleanup.
    """
    
    def __init__(self, fps: int = 5, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE):
        self.fps = fps
        self.size = size
        self.out = None
        
    def __enter__(self):
        """Initialize video writer."""
        os.makedirs("replay", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('replay/tmp.mp4', fourcc, self.fps, self.size)
        return self
        
    def write(self, pil_image: Image.Image):
        """Write a PIL image frame to video."""
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.out.write(frame)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up video resources and convert format."""
        if self.out:
            self.out.release()
            
        save_name = "replay/custom_record.mp4"
        if os.path.exists(save_name):
            os.remove(save_name)
            
        # Convert video format for better compatibility
        os.system(f"ffmpeg -i replay/tmp.mp4 -vcodec libx264 {save_name}")
        os.remove("replay/tmp.mp4")
        cv2.destroyAllWindows()