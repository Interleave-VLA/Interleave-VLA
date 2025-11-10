#!/bin/bash

#SBATCH --job-name=eval-bridge
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source .env

# hf download Interleave-VLA/interleave-pi0-bridge --include "step34799.pt"
ckpt_path="/path/to/open-pi-zero/logs/train/your_run_name/step34799.pt"

echo "Using checkpoint: $ckpt_path..."

TASKS=(
    # ================= Original Simpler-Env Tasks ==================
    # "widowx_carrot_on_plate"
    "widowx_put_eggplant_in_basket"
    # "widowx_spoon_on_towel"
    # "widowx_stack_cube"

    # ================= Curated Interleaved Simpler-Env Tasks ==================
    # "widowx_carrot_on_plate_unseen_lighting"
    # "widowx_spoon_on_towel_new_table_cloth"
    # "widowx_spoon_on_towel_google"
    # "widowx_redbull_on_plate"
    # "widowx_tennis_ball_in_basket"
    # "widowx_zucchini_on_towel"
    # "widowx_tape_measure_in_basket"
    # "widowx_toy_dinosaur_on_towel"
    # "widowx_stapler_on_paper"
)

N_EVAL_EPISODE=240   # octo simpler runs 3 seeds with 24 configs each, here we run 10 seeds

for TASK in ${TASKS[@]}; do
    # python -m debugpy --wait-for-client --listen 5680 \
    HYDRA_FULL_ERROR=1 python \
        scripts/run.py \
        --config-name=interleaved_bridge \
        --config-path=../config/eval \
        device=cuda:0 \
        seed=42 \
        n_eval_episode=$N_EVAL_EPISODE \
        n_video=$N_EVAL_EPISODE \
        env.task=$TASK \
        horizon_steps=4 \
        act_steps=4 \
        use_bf16=True \
        use_torch_compile=True \
        name=bridge_beta \
        checkpoint_path=$ckpt_path \
        env.adapter.dataset_statistics_path="$(pwd)/config/bridge_statistics.json" \
        # env.adapter.use_assets_from_google=True
done