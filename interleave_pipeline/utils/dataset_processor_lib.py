import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    Owlv2Processor, Owlv2ForObjectDetection, AutoTokenizer, AutoModelForCausalLM
)
import re
import os
import ast
import pickle
import logging
from .prompt_engineering import engineered_prompt

class DatasetProcessor:
    def __init__(self, model_paths, config, dataset_name, split='train'):
        self.path_to_owlv2 = model_paths['path_to_owlv2']
        self.path_to_qwen = model_paths['path_to_qwen']
        self.path_to_dataset = config[dataset_name]['path_to_dataset']
        self.path_to_save = config[dataset_name]['path_to_save']
        self.get_prompt = config[dataset_name]['get_prompt']
        self.get_observation = config[dataset_name]['get_observation']
        self.split = split
        self.save_count = {"train": 0, "val": 0}
        
        # Create necessary directories
        os.makedirs(self.path_to_save, exist_ok=True)
        os.makedirs(self.path_to_save + "/images", exist_ok=True)

    def initialize_models(self, device_map="cuda:0"):
        """Initialize all required models"""
        self.det_processor = Owlv2Processor.from_pretrained(self.path_to_owlv2)
        self.det_model = Owlv2ForObjectDetection.from_pretrained(self.path_to_owlv2, device_map=device_map)
        self.ll_tokenizer = AutoTokenizer.from_pretrained(self.path_to_qwen)
        self.ll_model = AutoModelForCausalLM.from_pretrained(self.path_to_qwen, device_map=device_map)
        
    def qwen_query(self, conversation):
        """Query the language model with a conversation"""
        text = self.ll_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.ll_tokenizer([text], return_tensors="pt").to(self.ll_model.device)
        generated_ids = self.ll_model.generate(
            **model_inputs,
            max_new_tokens=64
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.ll_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def qwen_query_objects(self, sentence):
        """Extract noun phrases representing objects from a sentence"""
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
        output = self.qwen_query(conversation)
        try:
            phrase_list = ast.literal_eval(output)
        except:
            return []
        if phrase_list and all(phrase in sentence for phrase in phrase_list):
            return list(set(phrase_list))  # unique
        return []
    
    def qwen_summarize_task(self, language_instruction):
        """In `utaustin mutex` dataset, we use this function to summarize the task into a single, short command."""
        conversation = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You should always reply in English."},
            {
                "role": "user",
                "content":
f"""Please simplify the following detailed instruction for a robotic task into a single, short command. The command should be phrased as a direct instruction to the robot, for example "put the yellow and white mug in the microwave and close it".
Here is the detailed instruction: '{language_instruction}'
Output:
"""
            }
        ]
        output = self.qwen_query(conversation)
        return output

    @staticmethod
    def sanitize_filename(filename):
        """Sanitize a filename by removing invalid characters"""
        return re.sub(r'[\\/*?:"<>|]', '_', filename)

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

    def process(self, images, prompt, data_to_save, split, file_id):
        """Process images with object detection and save results"""
        # prompt = self.qwen_summarize_task(prompt) # feature enabled in `utaustin mutex`
        for step in data_to_save['steps']:
            step['language_instruction'] = prompt
        # Try to extract objects from prompt
        for _ in range(3):  # retry 3 times
            fine_objs = self.qwen_query_objects(prompt)
            if fine_objs:
                break
                
        data_to_save['interleaved_prompt'] = {obj: None for obj in fine_objs}
        if not fine_objs:
            logging.warning(f"Fine objects are not found in `{prompt}`.")
            with open(f"{self.path_to_save}/{file_id}.pkl", "wb") as f:
                pickle.dump(data_to_save, f)
            return False
            
        text_labels, r_text_labels = [], {}
        for obj in fine_objs:
            _obj = engineered_prompt.get(obj, obj)
            text_labels.append(f"a photo of a {_obj}")
            r_text_labels[_obj] = obj
        
        # Process a subset of images
        for i in range(0, len(images), max(5, round(len(images) / 10))):
            # Confidence threshold for object detection; empirically set to maximize recall while encouraging predictions
            threshold = max(0.3 - i * 0.02, 0.1)
            image = images[i]
            image = Image.fromarray(image.numpy())
            try:
                inputs = self.det_processor(text=text_labels, images=image, return_tensors="pt")
            except:
                print("[ERROR] text labels is ", text_labels, " prompt is ", prompt)
                break
            inputs = inputs.to(self.det_model.device)

            with torch.inference_mode():
                outputs = self.det_model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.det_processor.post_process_object_detection(
                outputs=outputs, threshold=threshold, target_sizes=target_sizes
            )

            boxes, scores, labels = results[0]["boxes"].cpu(), results[0]["scores"].cpu(), results[0]["labels"].cpu()
            best_results = {}
            for box, score, label in zip(boxes, scores, labels):
                score = score.item()
                label = r_text_labels.get(text_labels[label.item()][len("a photo of a "):], '')
                box = [round(i) for i in box.tolist()]
                x_min, y_min, x_max, y_max = box
                box_width, box_height = x_max - x_min, y_max - y_min
                if box_width * box_height * 2 >= image.width * image.height:
                    continue  # we prefer smaller objects
                if label not in best_results or score > best_results[label]["score"]:
                    best_results[label] = {"box": box, "score": score}
                    
            if not best_results:
                continue
                
            logging.info(f"split = {split}, prompt = {prompt}, id = {file_id}, objs = {fine_objs}")
            
            for label, best_result in sorted(best_results.items(), key=lambda item: item[1]["score"], reverse=True):
                if label.lower() in prompt.lower() and data_to_save['interleaved_prompt'][label] is None:
                    expanded_box = self.expand_box(best_result['box'], image.width, image.height, expand_ratio=0.2)
                    obj = image.crop(expanded_box)
                    data_to_save['interleaved_prompt'][label] = np.array(obj)
                    obj.save(f"{self.path_to_save}/images/{file_id}__{self.sanitize_filename(label.replace(' ', '_'))}__conf{best_result['score']:.2f}.jpg")
                    
            if all(v is not None for v in data_to_save['interleaved_prompt'].values()):
                break
                
        # Save the processed data
        with open(f"{self.path_to_save}/{file_id}.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
            
        detected = any(v is not None for v in data_to_save['interleaved_prompt'].values())
        safe_prompt = self.sanitize_filename(prompt.replace(' ', '_'))
        
        if detected:
            image.save(f"{self.path_to_save}/images/{file_id}__obs__{safe_prompt}.jpg")
        else:
            image.save(f"{self.path_to_save}/images/[crop_error]{file_id}__obs__{safe_prompt}.jpg")
            
        return detected

    def set_save_count(self, split, count):
        """Set the save counter for a specific split"""
        self.save_count[split] = count

    def process_dataset(self, dataset, split):
        """Process a dataset with optional sharding"""
        success = 0
        
        from tqdm import tqdm
        
        pbar = tqdm(dataset, desc=f"Processing {split} episodes")
        for i, episode in enumerate(pbar):
            ep_to_save = {"steps": [step for step in episode['steps']]}
            images = [self.get_observation(step) for step in ep_to_save["steps"]]
            prompts = [self.get_prompt(step) for step in ep_to_save["steps"]]
            if not prompts[0]:
                continue
            # Group consecutive steps with identical prompts
            subsequences = []
            current_seq = {"steps": [], "images": [], "prompt": prompts[0]}
            for i, (step, image, prompt) in enumerate(zip(ep_to_save["steps"], images, prompts)):
                if i > 0 and prompt != current_seq["prompt"]:
                    # Start a new subsequence when prompt changes
                    subsequences.append(current_seq)
                    current_seq = {"steps": [], "images": [], "prompt": prompt}
                current_seq["steps"].append(step)
                current_seq["images"].append(image)
            
            # Add the last subsequence
            if current_seq["steps"]:
                subsequences.append(current_seq)
            
            # Process each subsequence separately
            for subseq_idx, subseq in enumerate(subsequences):
                sub_ep_to_save = {"steps": subseq["steps"]}
                subseq_file_id = f"{split}{self.save_count[split]:05d}_{subseq_idx:02d}"
                
                sub_success = self.process(
                    subseq["images"], 
                    subseq["prompt"], 
                    sub_ep_to_save, 
                    split,
                    file_id=subseq_file_id
                )
                success += sub_success
            
            self.save_count[split] += 1
            pbar.set_postfix({"success": success})
            
        return success