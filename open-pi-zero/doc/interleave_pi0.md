# Interleave-$\pi_0$

## Overview

This repository implements the interleaved version of the [$\pi_0$](https://www.physicalintelligence.company/blog/pi0) model from Physical Intelligence (Pi), referred to as Interleave-$\pi_0$. It extends the codebase from [open-pi-zero](https://github.com/allenzren/open-pi-zero).

## Installation

### Install Interleave-$\pi_0$

```shell
conda create -n pi0 python=3.10.16
conda activate pi0
# manually install the GPU build of PyTorch 2.5.1 (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd open-pi-zero
pip install -e .
```

### Install Simpler-Env

To evaluate Interleave-$\pi_0$ on Simpler-Env, please install our customized version of SimplerEnv, which includes additional out-of-domain evaluation tasks. The repository is available [here](https://github.com/Interleave-VLA/SimplerEnv). Follow the steps below to complete the installation:

```shell
git clone https://github.com/Interleave-VLA/SimplerEnv.git
cd SimplerEnv
pip install -e .

git submodule update --init --recursive
cd ManiSkill2_real2sim
pip install -e .
```

After installation, you can verify your setup by running the following command:
```shell
python -m sapien.example.offscreen
```
If an image is generated in the current directory, your installation is successful. Otherwise, refer to the [official installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) for more details. If you are using a remote headless server, issues may be due to missing or misconfigured rendering packages. If the problem persists after following the [troubleshooting guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting), you can try our Docker image [maniskill](https://hub.docker.com/r/alfayoung/maniskill-with-ssh).

## Training

### Training Data Preparation

You can generate the training data yourself to customize the pipeline and inspect data quality. Refer to the [interleaved dataset pipeline](/interleave_pipeline/README.md) for details.

Alternatively, you can download the prepared dataset from [Hugging Face](https://huggingface.co/collections/Interleave-VLA/interleave-vla-dataset-6866a10654b16d02032db7a1).

### Train Interleave-$\pi_0$

Configure your training parameters in [train_interleave.sh](/open-pi-zero/slurm/train_interleave.sh), then start training with:

```shell
bash open-pi-zero/slurm/train_interleave.sh
```

## Evaluation

### Evaluation on Simpler-Env

Evaluate Interleave-$\pi_0$ on Simpler-Env with:

```shell
bash open-pi-zero/slurm/eval_interleaved_simpler_bridge.sh
```

To compare performance, you can also evaluate the original (non-interleaved) $\pi_0$ with:

```shell
bash open-pi-zero/slurm/eval_simpler_bridge.sh
```
