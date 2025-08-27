#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G

source .env
export VLA_DATA_DIR=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/tensorflow_datasets/OXE_pretrain
export WANDB_MODE=offline

export WANDB__SERVICE_WAIT=300

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

export MASTER_ADDR="localhost"
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# run script with selected configuration using torchrun
HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run.py \
  --config-name=interleaved_pretrain \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_sampling=beta \
  use_torch_compile=True \
  use_bf16=True \
  use_amp=True

# debug script
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python -m debugpy --wait-for-client --listen 5680 \
#   scripts/run.py \
#   --config-name=interleaved_pretrain \
#   action_lr=0.00005 \
#   vlm_lr=0.00005 \
#   flow_sampling=beta \
#   use_torch_compile=False \
#   use_bf16=True \
#   use_amp=True \
#   'resume_checkpoint_path="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/pi0_suite/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt"' \
#   'resume_checkpoint_step=False';