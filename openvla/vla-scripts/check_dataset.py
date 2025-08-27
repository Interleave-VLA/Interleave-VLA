import os
os.sys.path.append("..")

from internvl.extern.hf.configuration_internvl import OpenVLAConfig
from internvl.extern.hf.processing_internvl import InternvlProcessor
from internvl.util.data_utils import PaddedCollatorForActionPrediction
from internvl.vla.action_tokenizer import ActionTokenizer

from internvl.vla.datasets import RLDSBatchTransform, RLDSDataset

from torch.utils.data import DataLoader

import tensorflow as tf

import pickle

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

'''
@tf.function
def func(x):
    # import pdb; pdb.set_trace()
    x = x + 1
    return x

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(func)
'''

# ============================= Params ===============================
# OXE
data_root = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/tensorflow_datasets"
dataset_name = "bridge_dataset"
# vima
# data_root = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/tensorflow_datasets"
# dataset_name = "vima_dataset"
vla_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/internvl_checkpoint/2b"
save_path = "batch_data.pkl"
# ====================================================================

config = OpenVLAConfig.from_pretrained(vla_path)
processor = InternvlProcessor(vla_path, config)
n_bins = 256
token_list = [f'<ACTION_{i}>' for i in range(1, n_bins + 1)]
processor.tokenizer.add_tokens(token_list, special_tokens=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)

batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor,
)
vla_dataset = RLDSDataset(
    data_root,
    dataset_name,
    batch_transform,
    shuffle_buffer_size=100_000,
    image_aug=False,
)

print(f"Finished loading dataset, dataset length = {len(vla_dataset)}")

collator = PaddedCollatorForActionPrediction(
    processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id
)
dataloader = DataLoader(
    vla_dataset,
    batch_size=128,
    sampler=None,
    collate_fn=collator,
    num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
)

print("Finished loading dataloader.")

for batch_idx, batch in enumerate(dataloader):
    print(batch)
    with open(save_path, "wb") as f:
        pickle.dump(batch, f)
    break