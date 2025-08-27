#!/bin/bash

# Please modify `utils/dataset_config.py` first, specify the path to the processed dataset and vision foundation model checkpoint

dataset_name=bridge_dataset

# bridge_dataset
# fractal20220817_data
# berkeley_autolab_ur5
# iamlab_cmu_pickup_insert_converted_externally_to_rlds
# stanford_hydra_dataset_converted_externally_to_rlds
# austin_sirius_dataset_converted_externally_to_rlds
# jaco_play
# ucsd_kitchen_dataset_converted_externally_to_rlds
# bc_z
# language_table
# utaustin_mutex

NUM_SHARDS=4
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    python post_process.py \
        --dataset-name $dataset_name \
        --shard-id $shard_index \
        --total-shards $NUM_SHARDS &
done
wait