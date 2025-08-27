total_num_actions=4000000
# Choose from the following datasets:
dataset_name="berkeley_autolab_ur5"
# berkeley_autolab_ur5
# fractal20220817_data
# jaco_play
# stanford_hydra_dataset_converted_externally_to_rlds
# bc_z
# droid
# berkeley_autolab_ur5
# iamlab_cmu_pickup_insert_converted_externally_to_rlds
# austin_sirius_dataset_converted_externally_to_rlds
# ucsd_kitchen_dataset_converted_externally_to_rlds
# bridge_dataset
# language_table
# utaustin_mutex

NUM_SHARDS=1
for split in train; do
    echo "Processing split: $split" 
    for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
        python pretrain_gen.py \
            --total_num_actions $total_num_actions \
            --dataset_name $dataset_name \
            --split $split \
            --shard_index $shard_index \
            --num_shards $NUM_SHARDS &
    done
    wait
done