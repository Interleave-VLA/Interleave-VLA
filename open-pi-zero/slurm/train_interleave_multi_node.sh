#!/bin/bash

source .env
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=300

export MASTER_ADDR=`echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`
export MASTER_PORT=21000
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=128
export NCCL_IB_HCA=mlx5_4 # mlx5_8
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export GPU_PER_DEVICE="$(nvidia-smi --list-gpus | wc -l)"
export NUM_MACHINES=$VC_TASK1_NUM
export NODE_RANK=$VC_TASK_INDEX

log_file="train_r$NODE_RANK.log"

# nvidia_log_file="nvidia_smi_r$NODE_RANK.log"
# nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 0.5 > $nvidia_log_file &
# MONITOR_PID=$!

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start Time: $current_time" | tee $log_file
echo "Num Machines $NUM_MACHINES" | tee -a $log_file
echo "Num GPUs $GPU_PER_DEVICE" | tee -a $log_file
echo "Master Address $MASTER_ADDR" | tee -a $log_file
echo "Master Port $MASTER_PORT" | tee -a $log_file
echo "Rank $NODE_RANK" | tee -a $log_file

# 启动分布式训练
HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=$NUM_MACHINES \
  --nproc_per_node=$GPU_PER_DEVICE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run.py \
  --config-name=interleaved_bridge \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_sampling=beta \
  use_torch_compile=True \
  use_bf16=True \
  use_amp=True \
  >> $log_file 2>&1

# kill $MONITOR_PID