#!/bin/bash

export VLA_LOG_DIR=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/pi0_suite/open-pi-zero/logs
export TRANSFORMERS_CACHE=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/pi0_suite/checkpoints

# widowx_carrot_on_plate
# "widowx_put_eggplant_in_basket"
# "widowx_spoon_on_towel"
# "widowx_stack_cube"
# "widowx_tape_measure_in_basket"
# google_robot_pick_eggplant

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -m debugpy --wait-for-client --listen 5680 \
scripts/interleaved_inference_in_simpler.py \
--task widowx_different_cube_on_towel \
--recording \
--use_bf16
