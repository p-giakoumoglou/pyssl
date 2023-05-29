# Self-Supervised Learning

Self-Supervised Learning (SSL) is a subfield of machine learning where the model learns to predict part of the data from other parts of the same data. This is often used in an unsupervised manner where the labels used for learning are generated from the data itself. This approach helps in creating representations of the data that can be used for downstream tasks such as classification, detection, and segmentation. SSL has been successful in a variety of domains such as computer vision, natural language processing, and reinforcement learning.

## 1. Project Structure

Project structure is as follows:

```bash
ssl/
├── models
│ ├── __init__.py
│ ├── byol.py
│ ├── simclr.py
│ ├── simsiam.py
│ ├── supcon.py
│ └── swav.py
├── networks
│ ├── __init__.py
│ ├── alexnet.py
│ ├── classifier.py
│ ├── densenet.py
│ ├── efficientnet.py
│ ├── resnet.py
│ └── vgg.py
├── optimizers
│ ├── __init__.py
│ ├── cosine_decay_warmup.py
│ ├── larc.py
│ ├── lars.py
│ └── lars_simclr.py
├── utils
├── main.py
├── main_linear.py
└── transformations.py
```

## 2. Methods

Currently, SimCLR, SupCon, and SimSiam are implemented. More methods to be added soon.

### 2.1 SimCLR

[SimCLR](https://arxiv.org/abs/2002.05709) (Simple Contrastive Learning of Visual Representations) is a framework for SSL of visual representations. It aims to learn representations by maximizing agreement between differently augmented views of the same data sample. 

Pre-training stage:

```bash
python main.py --model_name=simclr --backbone=resnet18 --batch_size=512 --optimizer=lars_simclr --weight_decay=0.0001 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=1.0 --final_lr=0 --num_epochs=1000
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=simclr --backbone=resnet18--batch_size=512 --optimizer=sgd_nesterov --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --base_lr=0.1 --final_lr=0 --num_epochs=100
```

### 2.2 SupCon

[SupCon](https://arxiv.org/abs/2004.11362) (Supervised Contrastive Learning) is another SSL method. While SimCLR uses a single positive pair (two augmented views of the same image) for each image in the batch, SupCon allows multiple positive pairs for each image. The key idea of SupCon is to use the labels available in supervised learning to define the positive and negative samples, which makes this method a hybrid of supervised and SSL.

### 2.3 SimSiam

[SimSiam](https://arxiv.org/abs/2011.10566) (Simple Siamese Networks) is a SSL approach that uses a simple framework to learn visual representations. The method involves two identical networks (hence the term "Siamese") that generate representations for two augmented views of the same image. The objective is to make these two representations as similar as possible. Unlike SimCLR and SupCon, SimSiam does not use negative pairs in its learning objective. This eliminates the necessity of large batch sizes and makes the method more computationally efficient.

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

### 2.4 BYOL

[BYOL](https://arxiv.org/abs/2006.07733) (Bootstrap Your Own Latent) is a novel SSL method that, like SimSiam, does not rely on negative pairs for training. Instead, it learns representations by comparing the representations of two augmented views of the same image. BYOL introduces a concept of a "target network" which is a moving average of the main network. The main network learns to align its representations with the slowly moving target network, which provides a form of consistency regularization.

### 2.5 SWAV

[SWAV](https://arxiv.org/abs/2006.09882) (Swapping Assignments between Views) is a SSL method that introduces a new way of assigning labels for the contrastive loss function. It uses a clustering algorithm to group similar representations together and assign them the same pseudo-label. SWAV introduces a unique "swap" operation where the assignments of the pseudo-labels are swapped between different views of the same image. This forces the model to learn consistent representations across different augmentations.

## 3. Results

In this section, we present the results of different SSL models.

| **Model** | **Backbone** | **Dataset** | **Acc@1** | **Acc@5** |
|-----------|--------------|-------------|-----------|-----------|
| SimSiam   | ResNet-18    | CIFAR-10    | 91.70%    | 99.62%    |
| SimCLR    | ResNet-18    | CIFAR-10    |           |           |
| SupCon    | ResNet-18    | CIFAR-10    |           |           |

### 3.1 SimSiam

Loss and learning rate during the pre-training phase of SimSiam:
![image](https://github.com/giakou4/ssl/assets/57758089/5f6010f5-dede-46a7-9ab1-586a9fa23f5e)

Loss and learning rate during fine-tuning linear classifier on top of frozen pre-trained backbone:
![image](https://github.com/giakou4/ssl/assets/57758089/197d52e5-9eda-477e-9fdb-58d89fdf5fa3)

Loss, top@1 accuracy, and top@5 accuracy during validation of frozen linear classifier and pre-trained backbone:
![image](https://github.com/giakou4/ssl/assets/57758089/1d9704ed-05e3-46fe-969c-0299478c4140)

### 3.2 SimCLR

### 3.3 SupCon
