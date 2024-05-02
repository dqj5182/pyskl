# Train and demo your action recognition videos as you want


## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
conda env create -f pyskl.yaml
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
You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}

# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```

For example,
```shell
# Training
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/posec3d/slowonly_r50_gym/joint.py 1 --validate --test-last --test-best

# Testing
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh configs/posec3d/slowonly_r50_gym/joint.py checkpoints/FineGYM/SlowOnly-R50/joint.pth 1 --eval top_k_accuracy mean_class_accuracy --out result.pkl
```

or without distributed,
```shell
# Training
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=30200 tools/train.py configs/posec3d/slowonly_r50_gym/joint.py --launcher pytorch --validate --test-last --test-best

# Testing
CUDA_VISIBLE_DEVICES=0 python tools/test.py --checkpoint checkpoints/FineGYM/SlowOnly-R50/joint.pth
```

## Acknowledgement
This code is mostly based on [PYSKL](https://github.com/kennymckormick/pyskl).