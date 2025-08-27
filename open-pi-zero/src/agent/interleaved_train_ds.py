"""
Main training agent using Accelerate. Using torch.compile and bfloat16 by default. Optionally (Q)LoRA.
"""

from functools import partial
import logging
import os
from collections import deque

import bitsandbytes as bnb
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.utils import set_seed

import wandb
from src.agent.interleaved_dataset import TorchRLDSInterleavedDataset
from src.agent.model_averaging import ModelAveraging
from src.model.vla.interleaved_pizero import InterleavedPiZero
from src.model.vla.interleaved_processing import InterleavedVLAProcessor
from src.utils.decorator import main_rank_only
from src.utils.metric import get_action_accuracy
from src.utils.monitor import (
    MainRankFilter,
    Timer,
    log_allocated_gpu_memory,
    log_execution_time,
)
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from src.agent.train import TrainAgent

log = logging.getLogger(__name__)

class InterleavedTrainAgent(TrainAgent):
    def __init__(self, cfg):
        # Accelerate setup
        self.cfg = cfg
        self.debug = cfg.get("debug", False)
        self.use_amp = cfg.get("use_amp", True)
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        
        # Initialize accelerator
        gradient_accumulation_steps = max(cfg.global_batch_size // cfg.per_device_batch_size // cfg.world_size, 1)        
        mixed_precision = 'bf16' if cfg.get("use_bf16", True) else 'fp16' if self.use_amp else 'no'
        # mixed precision is now handled in `accelerate`
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb" if cfg.get("wandb", False) else None
        )
        
        # Update device and main rank from accelerator
        self.device = self.accelerator.device
        self.main_rank = self.accelerator.is_main_process
        
        # Set seed for reproducibility
        if cfg.get("seed", None) is not None:
            set_seed(cfg.seed)
        
        # logging
        self.use_wandb = cfg.get("wandb", False) and self.main_rank
        if self.use_wandb:
            self.accelerator.init_trackers(
                project_name=cfg.wandb.project,
                init_kwargs={
                    "wandb": {
                        "entity": cfg.wandb.entity,
                        "name": cfg.wandb.run,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "id": self.wandb_id if hasattr(self, "wandb_id") else None,
                        "resume": "allow",  # not using resume_from
                    }
                }
            )
        
        log.addFilter(MainRankFilter(main_rank=self.main_rank))
        self.save_model_freq = int(cfg.save_model_freq)
        self.save_model_start = int(cfg.get("save_model_start", 0))
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm
        self.use_torch_compile = cfg.get("use_torch_compile", True)

        # model
        assert not ((cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights), (
            "Please load pretrained weights if quantizing VLM or using Lora."
        )
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )
            
        # Create model without DDP - Accelerate will handle distribution
        self.model = InterleavedPiZero(cfg, use_ddp=False)
        if cfg.resume_checkpoint_path:
            self.load_checkpoint(cfg.resume_checkpoint_path)
            if not cfg.resume_checkpoint_step:
                del self.cnt_batch
                del self.cnt_update
                del self.wandb_id
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        if cfg.freeze_vision_model:
            log.warning("Freezing vision model...")
            for name, param in self.model.vision_tower.named_parameters():
                param.requires_grad = False
                
        self.model.to(self.dtype)
        
        if self.use_torch_compile:
            self.model = torch.compile(
                self.model,
                mode="default",
            )
        
        # Determine batch size - Accelerate will handle gradient accumulation automatically
        self.grad_accumulation_steps = max(
            cfg.global_batch_size // cfg.per_device_batch_size // self.accelerator.num_processes, 1
        )
        actual_global_batch_size = (
            cfg.per_device_batch_size * self.grad_accumulation_steps * self.accelerator.num_processes
        )
        # dataloader
        self.train_dataloader = DataLoader(
            TorchRLDSInterleavedDataset(cfg.data.train, train=True).dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
            collate_fn=partial(self.collate_and_preprocess, split_mask=False, sample_fm_time=True),
        )
        self.run_eval = cfg.data.get("val", False)
        if self.run_eval:
            cfg_data_val = OmegaConf.merge(cfg.data.train, cfg.data.val)
            val_dataset = TorchRLDSInterleavedDataset(cfg_data_val, train=False).dataset
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.per_device_batch_size,
                pin_memory=True,
                collate_fn=partial(self.collate_and_preprocess, split_mask=True, sample_fm_time=False),
            )
            self.eval_thresholds = cfg.eval_thresholds
            self.eval_freq = cfg.eval_freq
            self.per_device_num_eval_batch = (
                cfg.eval_size // cfg.per_device_batch_size // self.accelerator.num_processes
            )
            
        log.info(f"Total length of dataset: {len(self.train_dataloader.dataset)}")
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {actual_global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        # optimizer setup
        self.train_vlm = cfg.train_vlm
        self.trained_parameters = self.model.action_expert_parameters
        self.action_optimizer = bnb.optim.AdamW8bit(
            self.model.action_expert_parameters,
            lr=cfg.action_lr,
            weight_decay=cfg.action_weight_decay,
        )
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.action_lr,
            min_lr=cfg.action_lr_scheduler.min_lr,
            warmup_steps=cfg.action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters (Action): {get_num_params_in_billions(self.action_optimizer):.3f}B"
        )
        
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = self.model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = self.model.trainable_vlm_parameters
            self.trained_parameters += vlm_trained_parameters
            self.vlm_optimizer = bnb.optim.AdamW8bit(
                vlm_trained_parameters,
                lr=cfg.vlm_lr,
                weight_decay=cfg.vlm_weight_decay,
            )
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            log.info(
                f"Number of trained parameters (VLM): {get_num_params_in_billions(self.vlm_optimizer):.3f}B"
            )
            
        if cfg.resume_checkpoint_path and cfg.resume_checkpoint_step:
            self.load_optimizer(cfg.resume_checkpoint_path)

        # Prepare everything with accelerator
        if self.train_vlm:
            self.model, self.action_optimizer, self.vlm_optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model, self.action_optimizer, self.vlm_optimizer, self.train_dataloader
            )
            if self.run_eval:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        else:
            self.model, self.action_optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model, self.action_optimizer, self.train_dataloader
            )
            if self.run_eval:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        ########### Input processing ###########
        # flow matching timestep sampling
        self.flow_sampling = cfg.get("flow_sampling", "beta")
        assert self.flow_sampling in [
            "uniform",
            "beta",
        ], f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        if self.flow_sampling == "beta":
            flow_alpha = cfg.get("flow_alpha", 1.5)
            flow_beta = cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        # processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = InterleavedVLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        assert self.processor.image_token_id == cfg.image_token_index, "Unexpected mismatch!"
        
        # Set up model averaging with unwrapped model
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.model_averaging = ModelAveraging(self.unwrapped_model, self.cfg, self.device)

    def preprocess_batch(self, batch, split_mask: bool, sample_fm_time: bool):
        images = batch["observation"]["image_primary"]
        proprios = batch["observation"]["proprio"]
        actions = batch["action"].squeeze(1)  # remove the time dimension
        texts = [
            text.decode("utf-8") for text in batch["task"]["interleaved_instruction"]["language_instruction"]
        ]
        texts = [text.replace("<image>", "<image_placeholder>") for text in texts]
        # === Check dataset ===
        instruction_images = [
            img[mask] for mask, img in zip(
                batch["task"]["interleaved_instruction"]["image_mask"], 
                batch["task"]["interleaved_instruction"]["image_mask"]
            )
        ]
        assert all(text.count("<image_placeholder>") == img.shape[0] for text, img in zip(texts, instruction_images))
        # =======
        interleaved_images = torch.stack([
            torch.cat(pair, axis=0)
            for pair in zip(images, batch["task"]["interleaved_instruction"]["image_instruction"])
        ], dim=0)
        # Reshape from B L H W C to B*L C H W
        batch_size = interleaved_images.shape[0]
        interleaved_images = einops.rearrange(interleaved_images, "B L H W C -> (B L) C H W")
        model_inputs = self.processor(text=texts, images=interleaved_images)
        # Reshape from B*L C H W to B L H W C
        model_inputs["pixel_values"] = einops.rearrange(
            model_inputs["pixel_values"],
            "(B L) C H W -> B L C H W", 
            B=batch_size
        )
        # Add a column of True values before each image mask for observation
        true_column = torch.ones((batch_size, 1), dtype=torch.bool)
        image_mask = torch.cat([true_column, batch["task"]["interleaved_instruction"]["image_mask"]], dim=1)
        # build causal mask and position ids for action
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.unwrapped_model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "image_mask": image_mask, 
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios.to(self.dtype),
            "actions": actions.to(self.dtype),
        }
        if split_mask:
            image_text_proprio_mask, action_mask = (
                self.unwrapped_model.split_full_mask_into_submasks(causal_mask)
            )
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
        else:
            inputs["causal_mask"] = causal_mask

        # sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

        return inputs
    
    def run(self):
        timer = Timer()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = 0 if not hasattr(self, "cnt_update") else self.cnt_update
        loss_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False

        # Get unwrapped model for direct access to methods
        self.model.train()

        # Main training loop
        self.accelerator.print("Starting training loop...")
        for epoch in range(self.n_updates // len(self.train_dataloader) + 1):
            for batch in self.train_dataloader:
                inputs = batch
                
                if self.debug and cnt_batch == 0:
                    images = batch["observation"]["image_primary"]
                    proprios = batch["observation"]["proprio"]
                    actions = batch["action"].squeeze(1)
                    texts = [
                        text.decode("utf-8")
                        for text in batch["task"]["interleaved_instruction"]["language_instruction"]
                    ]
                    log.info(f"Device {self.device}")
                    log.info(f"Texts {texts}")
                    log.info(f"Images {images.shape}")
                    log.info(
                        f"Actions {actions.shape} {actions.mean()} {actions.std()} {actions.max()} {actions.min()}"
                    )
                    log.info(
                        f"Proprios {proprios.shape} {proprios.mean()} {proprios.std()} {proprios.max()} {proprios.min()}"
                    )

                    # Save an image for debugging
                    if self.main_rank:
                        image = images[0, 0].clone().cpu()
                        image = Image.fromarray(image.numpy())
                        image.save(os.path.join(self.log_dir, f"image_debug.png"))

                # Forward pass
                loss = self.model(**inputs)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Step when accumulation is done
                if (cnt_batch + 1) % self.grad_accumulation_steps == 0:
                    # Clip gradients
                    self.accelerator.clip_grad_norm_(self.trained_parameters, self.max_grad_norm)
                    
                    # Step optimizers
                    self.action_optimizer.step()
                    self.action_lr_scheduler.step()
                    if self.train_vlm:
                        self.vlm_optimizer.step()
                        self.vlm_lr_scheduler.step()
                        
                    # Zero gradients
                    self.action_optimizer.zero_grad()
                    if self.train_vlm:
                        self.vlm_optimizer.zero_grad()
                        
                    # Update counter
                    cnt_update += 1
                    
                    # Model averaging updates
                    self.model_averaging.maybe_initialize(cnt_update)
                    self.model_averaging.maybe_update(cnt_update)
                    
                    # Save model checkpoint
                    if ((cnt_update % self.save_model_freq == 0 and cnt_update > self.save_model_start) 
                        or cnt_update == self.n_updates) and self.main_rank:
                        self.save_training(cnt_update, cnt_batch, main_rank=True)
                
                # Gather loss from all processes for logging
                loss_for_logging = self.accelerator.gather(loss)
                loss_deque.append(loss_for_logging.mean().item())
                
                # Run validation if needed
                if self.run_eval and (cnt_batch + 1) % self.eval_freq == 0:
                    self.accelerator.print(f"Running evaluation for {self.per_device_num_eval_batch} batches...")
                    new_eval_from_last_log = True
                    self.model.eval()
                    model_eval = self.model_averaging.get_model_module()
                    eval_accuracy = torch.zeros(len(self.eval_thresholds), device=self.device)
                    eval_l1_loss = torch.tensor(0.0, device=self.device)
                    
                    val_dataiterator = iter(self.val_dataloader)
                    with torch.inference_mode():
                        for _ in range(self.per_device_num_eval_batch):
                            try:
                                batch_eval = next(val_dataiterator)
                            except StopIteration:
                                val_dataiterator = iter(self.val_dataloader)
                                batch_eval = next(val_dataiterator)
                                
                            inputs = batch_eval
                            gt_actions = inputs.pop("actions")
                            preds = model_eval.infer_action(**inputs)
                            eval_accuracy += get_action_accuracy(gt_actions, preds, self.eval_thresholds)
                            eval_l1_loss += torch.nn.functional.l1_loss(preds, gt_actions)
                            
                    self.model.train()
                    
                    # Gather and average evaluation metrics
                    eval_accuracy = eval_accuracy / self.per_device_num_eval_batch
                    eval_l1_loss = eval_l1_loss / self.per_device_num_eval_batch
                    
                    # Reduce across processes
                    eval_accuracy = self.accelerator.gather_for_metrics(eval_accuracy)
                    eval_l1_loss = self.accelerator.gather_for_metrics(eval_l1_loss)
                    
                    if self.main_rank:
                        log_msg = f"Eval | l1 Loss: {eval_l1_loss.mean().item():.3f} | "
                        log_msg += " | ".join(
                            [
                                f"acc thres {threshold}: {accuracy.mean().item():.3f}"
                                for threshold, accuracy in zip(
                                    self.eval_thresholds, eval_accuracy
                                )
                            ]
                        )
                        self.accelerator.print(log_msg)
                
                # Log training progress
                if cnt_batch % self.log_freq == 0:
                    loss_metric = np.mean(loss_deque)
                    peak_vram = torch.cuda.max_memory_reserved() / (1024**3)
                    log_msg = f"Batch {cnt_batch} Update {cnt_update}: t {timer():8.4f} | vram {peak_vram:6.3f} | loss {loss_metric:6.4f} | action lr {self.action_optimizer.param_groups[0]['lr']:10.8f}"
                    if self.train_vlm:
                        log_msg += f" | vlm lr {self.vlm_optimizer.param_groups[0]['lr']:10.8f}"
                    self.accelerator.print(log_msg)
                    
                    if self.main_rank:
                        wandb_metrics = {
                            "loss - train": loss_metric,
                            "gradient steps": cnt_update,
                            "action lr": self.action_optimizer.param_groups[0]["lr"],
                        }
                        if self.train_vlm:
                            wandb_metrics["vlm lr"] = self.vlm_optimizer.param_groups[0]["lr"]
                        if new_eval_from_last_log:
                            for threshold, accuracy in zip(self.eval_thresholds, eval_accuracy):
                                wandb_metrics[f"eval acc - thres {threshold}"] = accuracy.mean().item()
                            wandb_metrics["eval l1 loss"] = eval_l1_loss.mean().item()
                            new_eval_from_last_log = False
                        
                        self.accelerator.log(wandb_metrics, step=cnt_batch)
                
                # Count batch and check for exit condition
                cnt_batch += 1
                if cnt_update >= self.n_updates:
                    return

    @log_execution_time(log)
    def save_training(self, cnt_update, cnt_batch, main_rank=True):
        """Save training checkpoint for model and optimizer"""
        if not main_rank:
            return
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"step_{cnt_update}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Get unwrapped model to save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        unwrapped_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer states
        optimizer_states = {
            "action_optimizer": self.action_optimizer.state_dict(),
            "action_scheduler": self.action_lr_scheduler.state_dict(),
        }
        if self.train_vlm:
            optimizer_states.update({
                "vlm_optimizer": self.vlm_optimizer.state_dict(),
                "vlm_scheduler": self.vlm_lr_scheduler.state_dict(),
            })
            
        # Save model averaging states
        optimizer_states.update({
            "model_averaging": self.model_averaging.state_dict(),
            "cnt_batch": cnt_batch,
            "cnt_update": cnt_update,
            "wandb_id": wandb.run.id if self.use_wandb else None,
        })
        
        torch.save(optimizer_states, os.path.join(checkpoint_path, "optimizer.pt"))
        log.info(f"Saved checkpoint at step {cnt_update}")
    
    @log_execution_time(log)
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        self.model.load_pretrained_weights(checkpoint_path)
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_states = torch.load(optimizer_path, map_location="cpu")
            self.cnt_batch = optimizer_states["cnt_batch"]
            self.cnt_update = optimizer_states["cnt_update"]
            self.wandb_id = optimizer_states.get("wandb_id", None)
            log.info(f"Loaded checkpoint: update {self.cnt_update}, batch {self.cnt_batch}")
    
    @log_execution_time(log)
    def load_optimizer(self, checkpoint_path):
        """Load optimizer state"""
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_states = torch.load(optimizer_path, map_location="cpu")
            self.action_optimizer.load_state_dict(optimizer_states["action_optimizer"])
            self.action_lr_scheduler.load_state_dict(optimizer_states["action_scheduler"])
            if self.train_vlm and "vlm_optimizer" in optimizer_states:
                self.vlm_optimizer.load_state_dict(optimizer_states["vlm_optimizer"])
                self.vlm_lr_scheduler.load_state_dict(optimizer_states["vlm_scheduler"])
            log.info("Loaded optimizer states")

    def collate_and_preprocess(self, batch, split_mask: bool, sample_fm_time: bool):
        # Custom collate function to handle different data types
        batch = self.custom_collate_fn(batch)
        # Preprocess batch
        inputs = self.preprocess_batch(
            batch,
            split_mask=split_mask,
            sample_fm_time=sample_fm_time,
        )        
        return inputs
    
    def custom_collate_fn(self, batch):
        if isinstance(batch[0], dict):
            result = {}
            for key in batch[0].keys():
                values = [sample[key] for sample in batch]
                if isinstance(values[0], dict):
                    result[key] = self.custom_collate_fn(values)
                elif isinstance(values[0], bytes):
                    result[key] = values
                else:
                    try:
                        result[key] = default_collate(values)
                    except:
                        result[key] = values
            return result
        elif isinstance(batch[0], (list, tuple)):
            transposed = zip(*batch)
            return [self.custom_collate_fn(samples) for samples in transposed]
        else:
            return default_collate(batch)