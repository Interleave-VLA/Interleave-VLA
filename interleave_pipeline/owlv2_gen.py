import tensorflow as tf
import tensorflow_datasets as tfds
import random
import argparse
import logging
import os
from utils.dataset_processor_lib import DatasetProcessor
from utils.dataset_config import config, model_path

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed and disable GPU for TensorFlow operations
random.seed(42)
tf.config.set_visible_devices([], 'GPU')

def main(args):
    """Main function to process the dataset"""
    # Create processor instance
    processor = DatasetProcessor(
        model_paths=model_path,
        config=config,
        dataset_name=args.dataset_name,
        split=args.split
    )
    
    # Initialize models with the specified GPU
    device_map = f"cuda:{args.shard_index}" if args.num_shards > 1 else "cuda:0"
    processor.initialize_models(device_map=device_map)
    
    # Load the dataset
    b = tfds.builder_from_directory(builder_dir=config[args.dataset_name]['path_to_dataset'])
    ds = b.as_dataset(split=args.split)
    total_length = len(ds)
    print(f"Total Dataset Length: {total_length}")
    
    # Handle sharding if specified
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)
        shard_size = total_length // args.num_shards
        processor.set_save_count(args.split, args.shard_index * shard_size)
        print(f"Sharding {args.shard_index + 1}/{args.num_shards}")
    
    # Skip specified number of examples if needed
    if args.num_to_skip > 0:
        print(f"Skipping {args.num_to_skip}...")
        ds = ds.skip(args.num_to_skip)
        processor.set_save_count(args.split, processor.save_count[args.split] + args.num_to_skip)
    
    print(f"Dataset length after preparation: {len(ds)}")
    ds.prefetch(1000)
    
    # Process the dataset
    success_count = processor.process_dataset(ds, args.split)
    print(f"Processing complete. Successfully processed {success_count} examples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset with sharding')
    parser.add_argument('--dataset_name', type=str, default='bridge_dataset', help='Name of the dataset to process')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val)')
    parser.add_argument('--shard_index', type=int, default=0, help='Index of the shard to process (0-based)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards')
    parser.add_argument('--num_to_skip', type=int, default=0, help='Number of examples to skip')
    
    args = parser.parse_args()
    main(args)