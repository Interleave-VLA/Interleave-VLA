#/bin/bash

# Please modify `utils/dataset_config.py` first, specify the path to the raw dataset and vision foundation model checkpoint

dataset_name="berkeley_autolab_ur5"

# bridge_dataset: train test
# fractal20220817_data: train
# berkeley_autolab_ur5: train test
# iamlab_cmu_pickup_insert_converted_externally_to_rlds: train
# stanford_hydra_dataset_converted_externally_to_rlds: train
# austin_sirius_dataset_converted_externally_to_rlds: train
# jaco_play: train test
# ucsd_kitchen_dataset_converted_externally_to_rlds: train
# bc_z: train val
# language_table: train
# utaustin_mutex: train

NUM_SHARDS=1
for split in train test; do
    echo "Processing split: $split" 
    for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
        python owlv2_gen.py \
            --dataset_name $dataset_name \
            --split $split \
            --shard_index $shard_index \
            --num_shards $NUM_SHARDS &
    done
    wait
done