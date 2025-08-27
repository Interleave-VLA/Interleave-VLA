from typing import Tuple, Any, Dict
import random
import ast
import torch
import cv2
import re
import glob
import os
import numpy as np
from PIL import Image
import logging
from transformers import (
    Owlv2Processor, Owlv2ForObjectDetection,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    Qwen2ForCausalLM, AutoTokenizer
)
from src.utils.geometry import euler2axangle, mat2euler, quat2mat
from src.agent.env_adapter.simpler import SimplerAdapter, BridgeSimplerAdapter
from src.model.vla.interleaved_processing import InterleavedVLAProcessor
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

class InterleavedSimplerAdapter(SimplerAdapter):
    IMAGE_PLACEHOLDER = "<image_placeholder>"
    prefered_objs = ["eggplant", "carrot", "tape measure", "toy dinosaur", "stapler"]
    
    def __init__(
        self,
        dataset_statistics_path: str,
        pretrained_model_path: str,
        tokenizer_padding: str,
        num_image_tokens: int,
        image_size: Tuple[int, int],
        max_seq_len: int,
        path_to_qwen: str,
        path_to_owlv2: str,
        use_assets_from_google: bool,
        google_assets_path: str = "",
        action_normalization_type: str = "bound",
        proprio_normalization_type: str = "bound",
    ):
        super().__init__(
            dataset_statistics_path,
            pretrained_model_path,
            tokenizer_padding,
            num_image_tokens,
            image_size,
            max_seq_len,
            action_normalization_type,
            proprio_normalization_type
        )
        self._use_assets_from_google = use_assets_from_google
        if self._use_assets_from_google:
            self._google_assets = self._get_google_assets(google_assets_path)
        else:
            self._tool_box = self._make_prompt_crop_tools(path_to_qwen, path_to_owlv2)
        self.processor = InterleavedVLAProcessor(
            self.tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def reset(self, instruction: str, env, obs: dict):
        image = get_image_from_maniskill2_obs_dict(env, obs)
        self._prompt, self._prompt_imgs = self._prepare_prompt(instruction, image)

    def _prepare_prompt(self, prompt: str, obs: np.ndarray):
        '''
        This function can be used to prepare the interleaved instruction for Simpler-Env.
        You may either hard-code the instruction or use the default case to automatically prepare the interleaved instruction.
        '''
        obs = Image.fromarray(obs)
        if self._use_assets_from_google:
            logging.info(f"using google assets...")
            cropped_images = {
                # "eggplant": self._google_assets["eggplant"][0],
                "carrot": self._google_assets["carrot"][0],
                # "green block": self._google_assets["green cube"][0],
                # "yellow block": self._google_assets["yellow cube"][0],
                # "spoon": self._google_assets["spoon"][0],
            }
            if len(cropped_images) > 1:
                selected_key = random.choice(list(cropped_images.keys()))
                cropped_images = {selected_key: cropped_images[selected_key]}
            for key in cropped_images.keys():
                cropped_images[key].save(f"{key}__conf1.jpg")
        elif "toy dinosaur" in prompt:
            obj_names = ["toy dinosaur"]
            cropped_images = self._det_crop_images(obs, ["green toy"], **self._tool_box)
            cropped_images["toy dinosaur"] = cropped_images["green toy"]
            del cropped_images["green toy"]
        elif "redbull can" in prompt:
            obj_names = ["redbull can"]
            cropped_images = self._det_crop_images(obs, ["blue and white can"], **self._tool_box)
            cropped_images["redbull can"] = cropped_images["blue and white can"]
            del cropped_images["blue and white can"]
        elif "plate with small sandwitches" in prompt:
            obj_names = ["plate with small sandwitches"]
            cropped_images = self._det_crop_images(obs, obj_names, **self._tool_box)
        elif "blue plate" in prompt:
            obj_names = ["blue plate"]
            cropped_images = self._det_crop_images(obs, obj_names, **self._tool_box)
        else: # default case, automatically prepare the interleaved instruction
            obj_names = self._qwen_query_objects(prompt, **self._tool_box)
            cropped_images = self._det_crop_images(obs, obj_names, **self._tool_box)
        obj_name_with_pos = []
        for obj_name in cropped_images.keys():
            match = re.search(re.escape(obj_name), prompt, flags=re.IGNORECASE)
            # replace one time only
            if match:
                obj_name_with_pos.append((match.start(), obj_name))
                prompt = re.sub(re.escape(obj_name), self.IMAGE_PLACEHOLDER, prompt, count=1, flags=re.IGNORECASE)
            # obj_name_with_pos.extend([(m.start(), obj_name) for m in re.finditer(re.escape(obj_name), prompt)])
            # prompt = prompt.replace(obj_name, self.IMAGE_PLACEHOLDER)
        obj_name_with_pos = sorted(obj_name_with_pos, key=lambda x: x[0])
        prompt_imgs = [cropped_images[obj_name] for _, obj_name in obj_name_with_pos]
        prompt_imgs = [img.resize(self.image_size) for img in prompt_imgs]
        prompt_imgs = [np.array(img) for img in prompt_imgs]
        return prompt, prompt_imgs
    
    def preprocess(self, env, obs: dict) -> dict:
        # assert self._prompt and self._prompt_imgs, "Call reset() first!"
        """using sxyz convention for euler angles"""
        image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        # no normalization for image before processor
        # always on cpu
        images = [image] + self._prompt_imgs
        for i in range(len(images)):
            images[i] = torch.as_tensor(images[i], dtype=torch.uint8).permute(2, 0, 1) # [3, H, W]
        images = torch.stack(images, dim=0) # [B, 3, H, W]
        
        model_inputs = self.processor(text=[self._prompt], images=images)

        # process proprio depending on the robot
        raw_proprio = self.preprocess_proprio(obs)

        # normalize proprios - gripper opening is normalized
        if self.proprio_normalization_type == "bound":
            proprio = self.normalize_bound(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["p01"]),
                np.array(self.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        elif self.proprio_normalization_type == "gaussian":
            proprio = self.normalize_gaussian(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        return {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "attention_mask": model_inputs["attention_mask"],
            "proprios": torch.as_tensor(proprio, dtype=torch.float32)[
                None, None
            ],  # [B, T, dim]
        }
    
    def _get_google_assets(self, google_asset_path):
        assets = {}
        asset_dirs = [d for d in os.listdir(google_asset_path) if os.path.isdir(os.path.join(google_asset_path, d))]
        for asset_dir in asset_dirs:
            logging.info(f"Enumerating {asset_dir}")
            asset_name = asset_dir.replace('_', ' ')  # Convert directory name to asset name
            paths = glob.glob(os.path.join(google_asset_path, asset_dir, '*'), recursive=False)
            assets[asset_name] = []
            for path in paths:
                assets[asset_name].append(Image.open(path).convert('RGB'))
        return assets
    
    def _get_qwenvl_and_processor(self, path_to_qwenvl):
        vl_processor = AutoProcessor.from_pretrained(path_to_qwenvl)
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path_to_qwenvl, device_map="auto")
        return dict(
            vl_model=vl_model, vl_processor=vl_processor
        )
    
    def _get_qwen_and_tokenizer(self, path_to_qwen):
            ll_tokenizer = AutoTokenizer.from_pretrained(path_to_qwen)
            ll_model = Qwen2ForCausalLM.from_pretrained(path_to_qwen, device_map="auto")
            return dict(
                ll_model=ll_model, ll_processor=ll_tokenizer
            )
    
    def _get_det_and_processor(self, path_to_owlv2):
        det_processor = Owlv2Processor.from_pretrained(path_to_owlv2)
        det_model = Owlv2ForObjectDetection.from_pretrained(path_to_owlv2, device_map="auto")
        return dict(
            det_model=det_model, det_processor=det_processor
        )
    
    def _make_prompt_crop_tools(self, path_to_qwen, path_to_owlv2):
        return dict(
            **self._get_qwen_and_tokenizer(path_to_qwen),
            **self._get_det_and_processor(path_to_owlv2)
        )
    
    def _qwen_query(self, conversation, ll_model, ll_tokenizer):
        text = ll_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = ll_tokenizer([text], return_tensors="pt").to(ll_model.device)
        generated_ids = ll_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = ll_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _qwen_query_objects(self, sentence, ll_model, ll_processor, **kwargs):
        conversation = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {
                "role": "user",
                "content": 
f"""Extract all noun phrases that represent objects or items from the following sentence. Include necessary adjectives, but ignore verbs, prepositions, directions and locations. Return the result in a list format.
Sentence: '{sentence}'
Output:
"""
            }
        ]
        output = self._qwen_query(conversation, ll_model, ll_processor)
        phrase_list = ast.literal_eval(output)
        if phrase_list and all(phrase in sentence for phrase in phrase_list):
            return list(set(phrase_list))
        else:
            return None

    def _expand_box(self, box, image_width, image_height, expand_ratio=0.1):
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

    def _det_crop_images(self, image: Image, obj_names, det_model, det_processor, **kwargs):
        texts = [[f"a photo of a {obj}" for obj in obj_names]]
        inputs = det_processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to(det_model.device)
        with torch.inference_mode():
            outputs = det_model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        result = det_processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)[0]
        boxes, scores, labels = result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()
        best_results = {}
        for box, score, label in zip(boxes, scores, labels):
            score_value = score.item()
            label_value = obj_names[label.item()] # name of the object
            print(label_value, ": ", score_value)
            x_min, y_min, x_max, y_max = box.tolist()
            box_width, box_height = x_max - x_min, y_max - y_min
            if label_value in self.prefered_objs:
                score_value += 0.5
            if box_width * box_height * 2 >= image.width * image.height:
                score_value -= 1 # we prefer smaller objects
            if label_value not in best_results or score_value > best_results[label_value]["score"]:
                best_results[label_value] = {"box": box, "score": score_value}
        ret = {}
        image.save(f"obs.jpg")
        # ================== Case 1: Save max conf only ============================
        max_key = max(best_results, key=lambda k: best_results[k]["score"])
        best_result = best_results[max_key]
        # max_key = random.choice(list(best_results.keys()))
        # best_result = best_results[max_key]
        box = [round(i) for i in best_result['box'].tolist()]
        obj = image.crop(self._expand_box(box, image.width, image.height, expand_ratio=0.2))
        obj.save(f"{max_key.replace(' ', '_')}__conf{best_result['score']:.2f}.jpg")
        ret[max_key] = obj
        # ================== Case 2: Save all ======================================
        # for label, best_result in best_results.items():
        #     box = [round(i) for i in best_result['box'].tolist()]
        #     obj = image.crop(box)
        #     obj.save(f"{label.replace(' ', '_')}__conf{best_result['score']:.2f}.jpg")
        #     ret[label] = obj
        return ret

class InterleavedBridgeSimplerAdapter(InterleavedSimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )
    def preprocess_proprio(self, obs: dict) -> np.array:
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio
    def postprocess_gripper(self, action: float) -> float:
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper