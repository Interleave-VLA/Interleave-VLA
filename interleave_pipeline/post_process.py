import argparse
from utils.dataset_postprocessor_lib import DatasetPostProcessor

def main(args):
    post_processor = DatasetPostProcessor(config_name=args.dataset_name)
    success, total = post_processor.process_dataset(shard_id=args.shard_id, total_shards=args.total_shards)
    print(f"Processing complete. Successfully processed {success} examples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    args = parser.parse_args()

    main(args)
