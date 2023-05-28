# Self-Supervised Learning

Self-Supervised Learning (SSL) is a subfield of machine learning where the model learns to predict part of the data from other parts of the same data. This is often used in an unsupervised manner where the labels used for learning are generated from the data itself. This approach helps in creating representations of the data that can be used for downstream tasks such as classification, detection, and segmentation. Self-Supervised Learning has been successful in a variety of domains such as computer vision, natural language processing, and reinforcement learning.

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
├── utils
├── main.py
├── main_linear.py
└── transformations.py
```

The **networks** folder contains Python files, each representing a different type of neural network backbone. These files include ```alexnet.py```, ```resnet.py```, ```vgg.py```, ```densenet.py```, and ```efficientnet.py```. Each file contains the implementation of the feature extractor (backbone) without the classifier used. In the ```__init__.py``` file, there is a dictionary named ```model_dict```. This dictionary serves as a central registry for all the models. Each key-value pair in the dictionary represents a backbone and its corresponding number of features as outputed from the last layer. For example, the key ```model_dict['alexnet']``` corresponds to the value ```[alexnet, 9216]```, where 'alexnet' is the model and '9216' is the number of features. Similarly, for ```'model_dict['resnet18']```, the corresponding value is ```[resnet18, 512]```, where 512 is the number of features​​.

The **models** folder contains Python files that implement different SSL models. These models include BYOL (```byol.py```), SimCLR (```simclr.py```), SimSiam (```simsiam.py```), SupCon (```supcon.py```), and SWAV (```swav.py```). Each file defines a class for the respective model that inherits from PyTorch's ```nn.Module``` class. The classes have an ```__init__()``` method for initialization and a ```forward()``` method to define the forward pass of the model returning the loss. Also, each file contains the a loss function implementing the corresponding loss used in each model.

The **utils** folder contains the implementation of various utilities used in this project.

The ```transformation.py``` file contains the definitions of the transformations applied to the images during the training and evaluation stages of the SSL models.

The ```main.py``` and ```main_linear.py``` files are the main scripts for training and evaluating SSL models. In ```main.py```, the pre-training of the backbone takes place, with t-SNE evaluation in the final epoch. In ```main_linear.py```, a linear classifier is trained on top of the frozen backbone to evaluate the representations of the SSL models.


## 2. Methods

### 2.1 SimCLR

[SimCLR](https://arxiv.org/abs/2002.05709) (Simple Contrastive Learning of Visual Representations) is a framework for self-supervised learning of visual representations. It aims to learn representations by maximizing agreement between differently augmented views of the same data sample. 

For SimCLR, run:

```bash
python main.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.3 --final_lr=0 --num_epochs=800
```

```bash
python main_linear.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=30
```

### 2.2 SupCon

[SupCon](https://arxiv.org/abs/2004.11362) (Supervised Contrastive Learning) is another self-supervised learning method. While SimCLR uses a single positive pair (two augmented views of the same image) for each image in the batch, SupCon allows multiple positive pairs for each image. The key idea of SupCon is to use the labels available in supervised learning to define the positive and negative samples, which makes this method a hybrid of supervised and self-supervised learning.

```bash
python main.py --model_name=supcon --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.3 --final_lr=0 --num_epochs=800
```

```bash
python main_linear.py --model_name=supcon --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=30
```

### 2.3 SimSiam

[SimSiam](https://arxiv.org/abs/2011.10566) (Simple Siamese Networks) is a self-supervised learning approach that uses a simple framework to learn visual representations. The method involves two identical networks (hence the term "Siamese") that generate representations for two augmented views of the same image. The objective is to make these two representations as similar as possible. Unlike SimCLR and SupCon, SimSiam does not use negative pairs in its learning objective. This eliminates the necessity of large batch sizes and makes the method more computationally efficient.

```bash
python main.py --model_name=simsiam --backbone=resnet18 --batch_size=512 --optimizer=sgd --weight_decay=0.0005 --momentum=0.9 --stop_at_epoch=800 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.03 --final_lr=0 --num_epochs=800
```

```bash
python main_linear.py --model_name=simsiam --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=100
```

### 2.4 BYOL

[BYOL](https://arxiv.org/abs/2006.07733) (Bootstrap Your Own Latent) is a novel self-supervised learning method that, like SimSiam, does not rely on negative pairs for training. Instead, it learns representations by comparing the representations of two augmented views of the same image. BYOL introduces a concept of a "target network" which is a moving average of the main network. The main network learns to align its representations with the slowly moving target network, which provides a form of consistency regularization.

### 2.5 SWAV

[SWAV](https://arxiv.org/abs/2006.09882) (Swapping Assignments between Views) is a self-supervised learning method that introduces a new way of assigning labels for the contrastive loss function. It uses a clustering algorithm to group similar representations together and assign them the same pseudo-label. SWAV introduces a unique "swap" operation where the assignments of the pseudo-labels are swapped between different views of the same image. This forces the model to learn consistent representations across different augmentations.

## 3 Running

Currently, SimCLR, SupCon, and SimSiam are implemented. More methods to be added soon.

### 3.1 SimCLR

Pre-training stage:

```bash
python main.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.3 --final_lr=0 --num_epochs=800
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=30
```

### 3.2 SupCon

Pre-training stage:

```bash
python main.py --model_name=supcon --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.3 --final_lr=0 --num_epochs=800
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=supcon --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=30
```

### 3.3 SimSiam

Pre-training stage:

```bash
python main.py --model_name=simsiam --backbone=resnet18 --batch_size=512 --optimizer=sgd --weight_decay=0.0005 --momentum=0.9 --stop_at_epoch=800 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.03 --final_lr=0 --num_epochs=800
```

Linear classifier fine-tuning:

```bash
python main_linear.py --model_name=simsiam --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=100
```
