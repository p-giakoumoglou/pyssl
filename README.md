# Self-Supervised Learning
A PyTorch implementation for the various Self-Supervised Learning (SSL) models.

## Dependencies

If you don't have python environment:

```
conda create -n ssl python=3.7
conda activate ssl
```

Then install the required packages:
```
pip install -r requirements.txt
```

## Project Structure

Project structure is as follows:

```bash
ssl/
├── models
│ ├── __init__.py
│ ├── byol.py
│ ├── simclr.py
│ ├── simsiam.py
│ └── supcon.py
├── networks
│ ├── __init__.py
│ ├── classifier.py
│ └── resnet.py
├── optimizers
│ ├── __init__.py
│ ├── larc.py
│ ├── lars.py
│ └── lars_simclr.py
├── utils
├── main.py
├── main_linear.py
├── transformations.py
└── utilities.py
```

## Models

### Run [SimCLR](https://arxiv.org/abs/2002.05709) (Simple Contrastive Learning of Visual Representations)

Pre-training stage:

```bash
python main.py --model_name=simclr --backbone=resnet18 --batch_size=512 --optimizer=lars_simclr --weight_decay=0.0001 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=1.0 --final_lr=0 --num_epochs=1000
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=simclr --backbone=resnet18 --batch_size=512 --optimizer=sgd_nesterov --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --base_lr=0.1 --final_lr=0 --num_epochs=100
```

### Run [SupCon](https://arxiv.org/abs/2004.11362) (Supervised Contrastive Learning)

TODO

### Run [SimSiam](https://arxiv.org/abs/2011.10566) (Simple Siamese Networks)

Pre-training stage:

```bash
python main.py --model_name=simsiam --backbone=resnet18 --batch_size=512 --optimizer=sgd --weight_decay=0.0005 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.03 --final_lr=0 --num_epochs=800
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=simsiam --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --base_lr=30 --final_lr=0 --num_epochs=100
```
or
```bash
python main_linear.py --model_name=simsiam --backbone=resnet18 --batch_size=256 --optimizer=lars --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --base_lr=0.02 --final_lr=0 --num_epochs=100
```

### Run [BYOL][(https://arxiv.org/abs/2006.07733) (Bootstrap Your Own Latent)

TODO

## Results

| **Model** | **Backbone** | **Dataset** | **Acc@1** | **Acc@5** |
|-----------|--------------|-------------|-----------|-----------|
| SimSiam   | ResNet-18    | CIFAR-10    | 91.70%    | 99.62%    |
| SimCLR    | ResNet-18    | CIFAR-10    | 85.35%    | 99.34%    |
| SupCon    | ResNet-18    | CIFAR-10    |           |           |
| BYOL      | ResNet-18    | CIFAR-10    |           |           |

## LOGS

- Added BYOL - still to implement scheduler for momentum
- Implemented base idea of DINO

## TODO

- Support Distributed Data Parallel (DDP)
- Support ImageNet
- Add DINO, SWAV, SimCLRv2
- Add MoCo, MoCov2, MoCov3
- Add InfoMin, InstDis, PIRL, CPC, CPCv2, CPCv2, LA, CMC

