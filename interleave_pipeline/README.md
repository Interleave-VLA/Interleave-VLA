# Training Data Generation

## Overview

![data_pipeline](/assets/data_pipeline.png)

To train Interleave-VLA, we curate an Interleaved X-Embodiment dataset with 210k robot manipulation trajectories from the Open X-Embodiment dataset using a streamlined three-step process: (1) use LLMs to extract key objects from instructions; (2) apply OWLv2 for open-vocabulary object detection and cropping; (3) use QwenVL to verify results and, when needed, refine segmentation with Segment Anything. The dataset spans diverse objects, tasks, and robot embodiments.

The pre-generated dataset is available on [Hugging Face](https://huggingface.co/collections/Interleave-VLA/interleave-vla-dataset-6866a10654b16d02032db7a1).

To regenerate the data or run on your own dataset partition, following sections are for you.

## Installation

Install SAM 2:
```shell
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Download the Open X-Embodiment dataset:
```bash
bash download_raw_openx.sh
```

## Pipeline

### Step1: Generate Coarse Version of Dataset

Run [step1](/interleave_pipeline/generate_step1_owlv2.sh) to generate the coarse version of the dataset using only OWLv2.

### Step2: Generate Fine Version of Dataset

Run [step2](/interleave_pipeline/generate_step2_sam2.sh) to generate the fine version of the dataset using Qwen2.5VL and SAM2.

### Step3: Convert to RLDS format

Go to the [tfds_dataset_builder/](/tfds_dataset_builder/) folder to convert the generated pickles into the RLDS format that both codebases ([Interleave-$\pi_0$](/open-pi-zero/) and [Interleave-OpenVLA](/openvla/)) are built upon. Note that this workflow is adapted from [this repository](https://github.com/moojink/rlds_dataset_builder).

Run `tfds build --overwrite` in the respective dataset folder to generate the RLDS dataset.

For example, to train on Bridge Data V2: go to the [tfds_dataset_builder/bridge_interleave/](/tfds_dataset_builder/bridge_interleave/) folder, edit the dataset path in `bridge_interleave.py`, and run `tfds build --overwrite` to generate the RLDS-format dataset.

🎉 Congratulations! You're ready to go. The generated dataset will be saved in the `~/tensorflow_datasets/` folder.