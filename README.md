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
bash tools/dist_train.sh configs/posec3d/slowonly_r50_gym/joint.py 2 --validate --test-last --test-best

# Testing
bash tools/dist_test.sh configs/posec3d/slowonly_r50_gym/joint.py checkpoints/FineGYM/SlowOnly-R50/joint.pth 1 --eval top_k_accuracy mean_class_accuracy --out result.pkl
```

or without distributed,
```shell
# Training
python train.py 

# Testing
python tools/test.py -checkpoint checkpoints/FineGYM/SlowOnly-R50/joint.pth --out result.pkl
```

## Citation

If you use PYSKL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry and the BibTex entry corresponding to the specific algorithm you used.

```BibTeX
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```

## Acknowledgement
This code is mostly based on [PYSKL](https://github.com/kennymckormick/pyskl).