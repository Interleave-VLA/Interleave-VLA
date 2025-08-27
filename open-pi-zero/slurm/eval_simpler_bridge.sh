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

ckpt_path="/path/to/open-pi-zero/logs/train/your_run_name/step19929.pt"

# better to run jobs for each task
TASKS=(
    # "widowx_carrot_on_plate"
    # "widowx_carrot_on_cool_plate"
    # "widowx_put_eggplant_in_basket"
    # "widowx_put_eggplant_in_basket_distractors"
    # "widowx_spoon_on_towel"
    # "widowx_zucchini_on_towel"
    # "widowx_spoon_on_towel_new_table_cloth"
    # "widowx_spoon_on_towel_google"
    # "widowx_dinosaur_on_towel"
    # "widowx_stack_cube"
    # "google_robot_pick_eggplant"
    "widowx_eggplant_on_plate"
)

N_EVAL_EPISODE=240   # octo simpler runs 3 seeds with 24 configs each, here we run 10 seeds

for TASK in ${TASKS[@]}; do
    # python -m debugpy --wait-for-client --listen 5680
    CUDA_VISIBLE_DEVICES=$1 HYDRA_FULL_ERROR=1 python \
        scripts/run.py \
        --config-name=bridge \
        --config-path=../config/eval \
        device=cuda:0 \
        seed=42 \
        n_eval_episode=$N_EVAL_EPISODE \
        n_video=$N_EVAL_EPISODE \
        env.task=$TASK \
        horizon_steps=4 \
        act_steps=4 \
        use_bf16=True \
        use_torch_compile=False \
        name=bridge_beta \
        checkpoint_path=$ckpt_path
        # time_max_period=100.0 \
        # action_expert_rope_theta=100.0
done
