import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.dataset_config import (
    config, post_process_config, model_path, post_process_censor_objects
)
from utils.prompt_engineering import engineered_prompt
from tqdm import tqdm
import os
import re
import pickle
import logging
import argparse
import glob
import random
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Callable, List, Optional

class DatasetPostProcessor:
    def __init__(self, config_name="bridge_dataset"):
        self.dataset_name = config_name
        self.path_to_qwenvl = model_path['path_to_qwenvl']
        self.path_to_sam2_cfg = model_path['path_to_sam2_cfg']
        self.path_to_sam2 = model_path['path_to_sam2']
        self.path_to_dataset = post_process_config[self.dataset_name]['path_to_dataset']
        self.path_to_save = post_process_config[self.dataset_name]['path_to_save']
        # self.assets_path = post_process_config[self.dataset_name]['assets_path']
        self.get_observation = config[self.dataset_name]['get_observation']
        self.get_prompt = config[self.dataset_name]['get_prompt']
        
        self.censor_objects = post_process_censor_objects[self.dataset_name]
        self.engineered_prompt = engineered_prompt
        
        # Create necessary directories
        os.makedirs(self.path_to_save, exist_ok=True)
        os.makedirs(self.path_to_save + "/images", exist_ok=True)
        
        # Stats for tracking success rate
        self.success = 0
        self.total = 0
        
        # Initialize models to None (will be loaded on demand)
        self.sam2_model = None
        self.sam2_predictor = None
        self.vl_processor = None
        self.vl_model = None
        
        # Load assets
        # self.assets = self.get_assets()

    def initialize_models(self, device_map=None):
        """Initialize all required models"""
        if device_map is None:
            device_map = "cuda:0"
        
        self.sam2_model = build_sam2(self.path_to_sam2_cfg, self.path_to_sam2).to(device_map)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.vl_processor = AutoProcessor.from_pretrained(self.path_to_qwenvl)
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.path_to_qwenvl, device_map=device_map
        )

    def get_assets(self):
        """Load asset images for all censored objects"""
        assets = {}
        for asset_name in self.censor_objects:
            paths = glob.glob(os.path.join(self.assets_path, asset_name.replace(' ', '_'), '*'), recursive=False)
            assets[asset_name] = []
            for path in paths:
                assets[asset_name].append(np.array(Image.open(path).convert('RGB')))
        return assets

    def get_random_asset(self, asset_name):
        """Get a random asset image for the given object name"""
        return random.choice(self.assets[asset_name])

    def qwen_check_label(self, image, label):
        """Check if the image matches the given label using the VL model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif image is None:
            return {
                "is_match": False,
                "explanation": "This is a padding image",
            }
            
        label = self.engineered_prompt.get(label, label)
        width, height = image.size
        scale = 224 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        _image = image.resize((new_width, new_height))
        
        # After some prompt engineering, we find that using Chinese prompt is better than English prompt.
        prompt = f"""忽略材质、真实性、用途和细节差异。图中物体的颜色和大致形状是否像"{label}"？
即使是玩具、模型或与标准形状稍有不同也算匹配。请主要关注颜色和大致轮廓，只要其中一个不符合，就不算匹配。
只需回答"Match:"或"No Match:"，然后简短说明颜色和大致形状的相似度。"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": _image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        inputs = self.vl_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.vl_model.device)
        
        with torch.inference_mode():
            output_ids = self.vl_model.generate(
                **inputs,
                max_new_tokens=64,
            )
        
        generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
        output_text = self.vl_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        result_text = output_text[0].strip()
        is_match = result_text.lower().startswith("match:")
        
        return {
            "is_match": is_match,
            "explanation": result_text,
        }

    @staticmethod
    def enlarge_image(image):
        """Resize an image while maintaining aspect ratio"""
        original_width, original_height = image.size
        scale = 1120 / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = image.resize((new_width, new_height))
        return resized_image

    @staticmethod
    def expand_box(box, image_width, image_height, expand_ratio=0.1):
        """Expand a bounding box by a given ratio"""
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min    
        x_expand = width * expand_ratio
        y_expand = height * expand_ratio
        new_x_min = max(0, x_min - x_expand)
        new_y_min = max(0, y_min - y_expand)
        new_x_max = min(image_width, x_max + x_expand)
        new_y_max = min(image_height, y_max + y_expand)
        return [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]

    def qwenvl_query_keypoint(self, image, label):
        """Query the VL model to detect keypoints for an object"""
        label = self.engineered_prompt.get(label, label)
        prompt = f'Detect the key point of the "{label}", return it in the form of points.'
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        inputs = self.vl_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.vl_model.device)

        with torch.inference_mode():
            output_ids = self.vl_model.generate(
                **inputs,
                max_new_tokens=64,
            )

        generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
        response = self.vl_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        try:
            element = ET.fromstring(response.strip())
            if element.tag == "points":
                x1 = int(element.attrib['x1'])
                y1 = int(element.attrib['y1'])
                return {
                    "found": True,
                    "point_2d": [x1, y1],
                }
        except ET.ParseError:
            pass

        json_match = re.search(r'({.*?})', response.replace('\n', ''))
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if "point_2d" in result:
                    return {
                        "found": True,
                        "point_2d": result["point_2d"]
                    }
            except json.JSONDecodeError:
                pass

        return {
            "found": False,
            "point_2d": response
        }

    def crop_image_from_obs(self, images, obj_name):
        """Crop object from observation images using SAM2"""
        for i in range(0, len(images), max(50, round(len(images) / 10))):
            image = Image.fromarray(images[i].numpy())
            image = self.enlarge_image(image)
            
            result = self.qwenvl_query_keypoint(image, obj_name)
            if not result['found']:
                continue
            
            self.sam2_predictor.set_image(image)
            input_point = np.array([result['point_2d']])
            input_label = np.array([1]) # foreground point
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            
            nonzero_indices = np.argwhere(masks[0] != 0)
            min_x = nonzero_indices[:, 1].min().item()
            max_x = nonzero_indices[:, 1].max().item()
            min_y = nonzero_indices[:, 0].min().item()
            max_y = nonzero_indices[:, 0].max().item()
            bbox = [min_x, min_y, max_x, max_y]
            expanded_box = self.expand_box(bbox, image.width, image.height, expand_ratio=0.2)
            obj = image.crop(expanded_box).resize((224, 224))
            return np.array(obj)
        return None

    def post_process_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process dictionary data to fix object detection issues"""
        for obj_name, cropped_image in data['interleaved_prompt'].items():
            if cropped_image is None:
                res = self.crop_image_from_obs(
                    [self.get_observation(step) for step in data["steps"]],
                    obj_name
                )
                self.success += (res is not None)
                self.total += 1
                data['interleaved_prompt'][obj_name] = res
                continue
            
            # for template in self.censor_objects:
            #     if template in obj_name.lower():
            result = self.qwen_check_label(cropped_image, obj_name)
            if not result['is_match']:
                res = self.crop_image_from_obs(
                    [self.get_observation(step) for step in data["steps"]],
                    obj_name
                )
                self.success += (res is not None)
                self.total += 1
                data['interleaved_prompt'][obj_name] = res
            # break
        
        # Filter out None values
        data['interleaved_prompt'] = {
            key: value for key, value in data['interleaved_prompt'].items() if value is not None
        }
        return data

    @staticmethod
    def sanitize_filename(filename):
        """Sanitize a filename by removing invalid characters"""
        return re.sub(r'[\\/*?:"<>| ]', '_', filename)

    def process_pkl_file(self, file_path: str) -> None:
        """Process a single pickle file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        processed_data = self.post_process_dict(data)
        filename = os.path.basename(file_path)
        
        # Save error image if no objects detected
        obs = Image.fromarray(self.get_observation(processed_data["steps"][0]).numpy())
        instruction = self.get_prompt(processed_data['steps'][0])
        if not processed_data['interleaved_prompt']:
            obs_filename = f"{self.path_to_save}/images/[crop_error]{filename[: -len('.pkl')]}__obs__{self.sanitize_filename(instruction)}.jpg"
        else:
            obs_filename = f"{self.path_to_save}/images/{filename[: -len('.pkl')]}__obs__{self.sanitize_filename(instruction)}.jpg"
        obs.save(obs_filename)
        
        # Save detected object images
        for key, value in processed_data['interleaved_prompt'].items():
            img = Image.fromarray(value).convert('RGB')
            img.save(f"{self.path_to_save}/images/{filename[: -len('.pkl')]}__{self.sanitize_filename(key)}.jpg")
        
        # Save processed data
        destination_path = os.path.join(self.path_to_save, filename)
        with open(destination_path, 'wb') as f:
            pickle.dump(processed_data, f)

    @staticmethod
    def get_shard_files(all_files: List[str], shard_id: int, total_shards: int) -> List[str]:
        """Get files for the specified shard"""
        return [f for i, f in enumerate(all_files) if i % total_shards == shard_id]

    def process_dataset(self, shard_id=0, total_shards=1):
        """Process the entire dataset with optional sharding"""
        # Initialize models with appropriate device mapping
        self.initialize_models(device_map=f"cuda:{shard_id}")
        
        # Find all pickle files
        all_files = glob.glob(os.path.join(self.path_to_dataset, "*.pkl"), recursive=False)
        if not all_files:
            print(f"No pkl found in {self.path_to_dataset}")
            return
        
        print(f"Found {len(all_files)} pkl files")
        if shard_id >= total_shards:
            raise ValueError("Shard ID must be less than total shards")
            
        # Get files for this shard
        files_to_process = self.get_shard_files(all_files, shard_id, total_shards)
        print(f"Current shard: {shard_id + 1}/{total_shards}; Will process {len(files_to_process)} files")
        
        # Process files with progress bar
        pbar = tqdm(files_to_process, desc="Processing episodes")
        for file_path in pbar:
            self.process_pkl_file(file_path)
            pbar.set_postfix({"success": f"{self.success} / {self.total}"})
        
        return self.success, self.total