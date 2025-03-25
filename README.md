# I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders

This code is the official implementation of [Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders](https://arxiv.org/abs/2503.18878).

The full pipeline will be released no later than 26.03.2025. The `README.md` will be updated accordingly. 

## Installation

1. Create a virtual environment and activate it (e.g conda environment):
```
conda create -n sae_reasoning python=3.11
conda activate sae_reasoning
```
2. Install Pytorch and torchvision following the official instructions:
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
3. Install build requirements:
```
pip install -r requirements.txt
```
4. Install our modified `sae_lens`:
```
cd sae_lens
pip install -e .
```
