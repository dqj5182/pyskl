# Action Recognition
Train and demo your action recognition videos as you want

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
conda env create -f environment.yaml
conda activate pyskl
pip install -e .
```

## Data Preparation
For GYM dataset, please download 2D pose annotations:
```shell
mkdir data
cd data
mkdir gym
cd gym
wget https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl
```

## Training & Testing
You can use following commands for training and testing.
```shell
# Training
CUDA_VISIBLE_DEVICES=0 python main/train.py

# Testing
CUDA_VISIBLE_DEVICES=2 python main/test.py --checkpoint checkpoints/FineGYM/SlowOnly-R50/joint.pth
```

## Acknowledgement
This code is mostly based on [PYSKL](https://github.com/kennymckormick/pyskl).