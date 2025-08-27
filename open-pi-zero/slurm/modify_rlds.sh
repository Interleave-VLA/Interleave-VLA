#!/bin/bash

#SBATCH --job-name=modify-rlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --time=23:59:59
#SBATCH --job-name=rlds
#SBATCH --output=logs/rlds-%J.log
#SBATCH --error=logs/rlds-%J.err

export VLA_DATA_DIR=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/OXE

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files in /tmp to store dataset during conversion
ulimit -n 20000

# dataset: bridge_dataset, or fractal20220817_data
python scripts/data/modify_rlds_dataset.py \
    --dataset=bridge_dataset \
    --data_dir=$VLA_DATA_DIR \
    --target_dir=$VLA_DATA_DIR/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=100 \
    --max_episodes_in_memory=400
