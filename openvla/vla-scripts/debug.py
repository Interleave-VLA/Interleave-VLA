"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
import accelerate
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
# from transformers import AutoModelForVision2Seq, BitsAndBytesConfig, ChameleonProcessor
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

import wandb
import sys
sys.path.append('/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/sunyihang/Chameleon-VLA/openvla')
from internvl.extern.hf.conversation import Conversation as InternvlPromptBuilder, get_conv_template
from internvl.extern.hf.processing_internvl import InternvlProcessor
from internvl.extern.hf.configuration_internvl import OpenVLAConfig
from internvl.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from internvl.util.data_utils import PaddedCollatorForActionPrediction
from internvl.extern.hf.modeling_internvl import OpenVLAForActionPrediction
from internvl.vla.datasets import (RLDSBatchTransform, RLDSDataset)

import atexit
import logging

logging.basicConfig(level=logging.INFO)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

resume_step = 0

@dataclass
class FinetuneConfig:
    # @formatter:off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    warmup_ratio: float = 0.03
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_backbone_lora: bool = False                                 # Set the LoRA adapter rank for the ViT
    use_llm_lora: bool = False                                      # Set the LoRA adapter rank for the LLM
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Internvl Arguments
    force_image_size: Optional[int] = 448                           # Base image size for Internvl Processor

    # Tracking Parameters
    exp_name: str = None
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "alfayoung2004-shanghai-jiao-tong-university" # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # @formatter :on


# @draccus.wrap()
def finetune() -> None:
    cfg = FinetuneConfig()
    cfg.data_root_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/sunyihang/Chameleon-VLA/dataset'
    cfg.dataset_name = 'se2_task1'
    cfg.batch_size = 8
    cfg.image_aug = True
    device = 'cuda:0'
    cfg.vla_path = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/sunyihang/Chameleon-VLA/pretrained'
    run_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/sunyihang/Chameleon-VLA/openvla/vla-scripts/run'
    
    print('###################')
    print(cfg.wandb_entity, cfg.wandb_project)

    accelerator = accelerate.Accelerator(split_batches=True)
    # accelerator = accelerate.Accelerator()
    device = accelerator.device

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    # torch.cuda.set_device(device)
    # torch.cuda.empty_cache()

    # ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
    # cfg.grad_accumulation_steps = ds_config["gradient_accumulation_steps"]
    # cfg.batch_size = torch.cuda.device_count() * ds_config["train_micro_batch_size_per_gpu"]
    # cfg.batch_size = ds_config["train_micro_batch_size_per_gpu"]

    # Configure Unique Experiment ID & Log Directory
    # exp_id = (
    #     f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    #     f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    #     f"+lr-{cfg.learning_rate}"
    # )
    # if cfg.use_backbone_lora or cfg.use_llm_lora:
    #     exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    # if cfg.use_quantization:
    #     exp_id += "+q-4bit"
    # if cfg.run_id_note is not None:
    #     exp_id += f"--{cfg.run_id_note}"
    # if cfg.image_aug:
    #     exp_id += "--image_aug"
    # if cfg.exp_name:
    #     exp_id = cfg.exp_name

    # # Start =>> Build Directories
    # run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    # save_dir = adapter_dir if cfg.use_backbone_lora or cfg.use_llm_lora else run_dir
    # os.makedirs(run_dir, exist_ok=True)

    # # Quantization Config =>> only if LoRA fine-tuning
    # quantization_config = None
    # if cfg.use_quantization:
    #     assert cfg.use_backbone_lora or cfg.use_llm_lora, "Quantized training only supported for LoRA fine-tuning!"
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
    #     )

    # Load OpenVLA Processor and Model using HF AutoClasses
    # config = OpenVLAConfig.from_pretrained(cfg.vla_path)
    # vla = OpenVLAForActionPrediction.from_pretrained(
    #     cfg.vla_path,
    #     torch_dtype=torch.bfloat16,
    #     config=config,
    #     quantization_config=quantization_config,
    #     local_files_only=True,
    # )
    # processor = InternvlProcessor(cfg.vla_path, config)
    # vla.img_context_token_id = processor.tokenizer.convert_tokens_to_ids(processor.IMG_CONTEXT_TOKEN)
    
    # TODO: we unfreeze the vision encoder weights temporarily
    # assert vla.config.vision_config.image_size == cfg.force_image_size, "Image size mismatch!"

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    # if cfg.use_quantization:
        # vla = prepare_model_for_kbit_training(vla)
    # else:
        # vla = vla.to(device)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    # if cfg.use_llm_lora:
    #     # vla.wrap_llm_lora(r=cfg.lora_rank, lora_alpha=2 * cfg.use_llm_lora)
    #     vla.wrap_llm_lora(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout)
    #     vla.config.use_llm_lora = cfg.use_llm_lora
    #     vla.language_model.print_trainable_parameters()

    # if cfg.use_backbone_lora:
    #     # vla.wrap_backbone_lora(r=cfg.use_backbone_lora, lora_alpha=2 * cfg.use_backbone_lora)
    #     vla.wrap_backbone_lora(r=cfg.use_backbone_lora, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout)
    #     vla.config.use_backbone_lora = cfg.use_backbone_lora
    #     vla.vision_model.print_trainable_parameters()

    # Create Action Tokenizerm
    n_bins = 256
    # token_list = [f'<ACTION_{i}>' for i in range(1, n_bins + 1)]
    # num_new_tokens = processor.tokenizer.add_tokens(token_list, special_tokens=True)
    # if num_new_tokens > 0:
    #     # TODO: input embedding == output embedding? are they both trained?
    #     # language_model.model.tok_embeddings.weight <= this is input embedding layer
    #     # vla.language_model.resize_token_embeddings(len(processor.tokenizer))
    #     output_embeddings = vla.language_model.get_output_embeddings().weight.data
    #     # === Average Init ===
    #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    #     output_embeddings[-num_new_tokens:] = output_embeddings_avg
    #     # === Linear Init ===
    #     # linear_init = torch.linspace(-1, 1, n_bins, device=device).view(-1, 1)
    #     # output_embeddings[-num_new_tokens:] = linear_init.expand(-1, output_embeddings.size(1))
    #     # === Random Init ===
    #     # output_embeddings[-num_new_tokens:] = torch.randn(num_new_tokens, output_embeddings.size(1), device=device) * 0.02
    #     vla.language_model.get_output_embeddings().weight.requires_grad = True
    #     vla.config.llm_config.vocab_size = len(processor.tokenizer)
    #     vla.language_model.config.vocab_size = len(processor.tokenizer)

    # action_tokenizer = ActionTokenizer(processor.tokenizer, bins=n_bins)

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    # trainable_params = [param for param in vla.parameters() if param.requires_grad]
    # optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    # num_warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
    # =========== Cosine Schedule =============
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, cfg.max_steps)
    # =========== Constant Schedule ===========
    # scheduler = get_constant_schedule(optimizer)
    
    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # batch_transform = RLDSBatchTransform(
    #     action_tokenizer,
    #     processor
    # )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        lambda x:x,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug
    )
    logging.info(f"dataset size = {len(vla_dataset)}")

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # if accelerator.is_main_process:
    #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    # Pad to left: https://huggingface.co/docs/transformers/main/en/model_doc/chameleon#usage-tips
    collator = PaddedCollatorForActionPrediction(1000)
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # setattr(vla.config, "hidden_size", vla.config.llm_config.hidden_size) # Fixme: Is this correct?
    # vla, dataloader, optimizer, scheduler = accelerator.prepare(vla, dataloader, optimizer, scheduler)

    # Initialize Logging =>> W&B
    # if accelerator.is_main_process:
        # atexit.register(lambda: wandb.finish() if accelerator.is_main_process else None)
        # wandb.login(key=os.environ["WANDB_API_KEY"])
        # wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    # recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    # recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    # recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps) as progress:
        # vla.train()
        # optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                    image_flags=batch["image_flags"]
                )
                loss = output.loss
            return
            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            accelerator.backward(normalized_loss)

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1 :].to(action_preds.device)
            assert action_preds.shape == action_gt.shape, "Action Prediction Shape Mismatch!"
            mask = torch.isin(action_gt, torch.tensor(action_tokenizer.token_ids, device=action_gt.device))

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            metrics_to_log = {
                "train_loss": smoothened_loss,
                "action_accuracy": smoothened_action_accuracy,
                "l1_loss": smoothened_l1_loss,
                "learning_rate": optimizer.param_groups[0]["lr"]
            }
            # Push Metrics to W&B (every 10 gradient steps)
            if accelerator.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(metrics_to_log, step=gradient_step_idx + resume_step)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() # cosine annealing
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    progress.set_postfix(metrics_to_log)
                    progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if cfg.save_latest_checkpoint_only:
                    checkpoint_dir = run_dir
                else:
                    checkpoint_dir = Path(str(run_dir)) / f"{gradient_step_idx + resume_step}_chkpt"
                    if accelerator.is_main_process:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                unwrapped_vla = accelerator.unwrap_model(vla)
                unwrapped_vla.save_pretrained(
                    checkpoint_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(vla)
                )
                processor.tokenizer.save_pretrained(checkpoint_dir)
                logging.info(f"Saved checkpoint at {checkpoint_dir} for process {accelerator.process_index} in gradient step {gradient_step_idx + resume_step}!")
                accelerator.wait_for_everyone()

            # Stop training when max_steps is reached
            if gradient_step_idx + resume_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    unwrapped_vla = accelerator.unwrap_model(vla)
    unwrapped_vla.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(vla)
    )
    unwrapped_vla = unwrapped_vla.merge_and_unload()
    print("Successfully merged weights...")
    if cfg.save_latest_checkpoint_only:
        # Overwrite latest checkpoint
        unwrapped_vla.save_pretrained(
            run_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(vla)
        )
        processor.tokenizer.save_pretrained(run_dir)
    else:
        # Prepare to save checkpoint in new directory
        checkpoint_dir = Path(str(run_dir)) / f"{gradient_step_idx + resume_step}_chkpt"
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Save dataset statistics to new directory
            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

        # Save processor and model weights to new directory
        unwrapped_vla.save_pretrained(
            checkpoint_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(vla)
        )
        processor.tokenizer.save_pretrained(checkpoint_dir)

if __name__ == "__main__":
    finetune()
