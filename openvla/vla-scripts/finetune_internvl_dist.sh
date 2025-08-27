export PYTHONPATH=".."
export WANDB_MODE=offline
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8


# ========================= RUN VIMA ========================================
accelerate launch \
--main_process_port 29500 \
--config_file accelerate/config.yaml \
finetune_internvl.py \
--vla_path internvl_checkpoint/2b \
--data_root_dir ~/tensorflow_datasets \
--dataset_name vima_interleave \
--run_root_dir runs \
--adapter_tmp_dir checkpoints \
--learning_rate 2e-5 \
--use_llm_lora False \
--use_backbone_lora False \
--image_aug False \
--max_steps 190000 \
--save_steps 19000 \
--exp_name "20250224 vima_interleave" \
--save_latest_checkpoint_only False
