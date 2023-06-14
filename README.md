![pyssl](https://github.com/giakou4/pyssl/assets/57758089/1ed62627-93ec-48ff-b80e-0cc61f6c2be7)

# PySSL

A PyTorch implementation of Self-Supervised Learning (SSL) methods

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/pyssl/LICENSE)
![stars](https://img.shields.io/github/stars/giakou4/pyssl.svg)
![issues-open](https://img.shields.io/github/issues/giakou4/pyssl.svg)
![issues-closed](https://img.shields.io/github/issues-closed/giakou4/pyssl.svg)
![size](https://img.shields.io/github/languages/code-size/giakou4/pyssl)


## 1. Prerequisites

Before proceeding, create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```shell
conda create -n pyssl
```
   
Activate the newly created environment:

```shell
conda activate pyssl
```

Once the environment is activated, install the required packages from the "requirements.txt" file using the following command:

```shell
pip install -r requirements.txt
```

## 2. Methods:

### 2.1 Barlow Twins

Barlow Twins is a self-supervised learning method that aims to learn representations by making the outputs of a neural network to be as similar as possible for two augmented views of the same image, while reducing the redundancy between the output components.

### 2.2 BYOL

BYOL (Bootstrap Your Own Latent) is an SSL method that learns representations by predicting one view of an input from another view of the same input, without the necessity for negative pairs which is the case for most contrastive learning methods.

### 2.3 DINO

DINO (Distillation of Self-supervised Networks) is a method that trains a student network using the outputs of a teacher network, where the teacher network is an exponentially moving average of the student network. The key idea is to use distillation loss to capture information from different viewpoints of the same image.

### 2.4 MOCO, MOCOv2, MOCOv3

MOCO (Momentum Contrast) series are methods that utilize a dynamic dictionary implemented with a queue and a moving-averaged encoder. The methods aim to maximize similarity between a query and its positive key and minimize similarity between the query and negative keys (v2 and v3 are updated versions with improvements over the original).

### 2.5 SimCLR, SimCLRv2

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a method that utilizes a contrastive loss function to distinguish between similar and dissimilar images. The v2 version introduces a simple method for augmenting the data sample with a learnable nonlinear transformation.

### 2.6 SimSiam

SimSiam (Simple Siamese) aims to learn representations by predicting one view of an input from another view of the same input, similar to BYOL. However, unlike BYOL, SimSiam does not use batch normalization in the prediction MLP, and does not require the use of a momentum encoder or moving average.

### 2.7 SupCon

SupCon (Supervised Contrastive Learning) is a method that uses a contrastive loss function but also makes use of labels, if available, to further refine the learning of representations by encouraging dissimilarities between representations of different classes.

### 2.8 SwAV

SwAV (Swapping Assignments between multiple Views of the same image) is a clustering-based method which learns representations by swapping the cluster assignments between different views of the same image, and then minimizing the difference between the swapped assignments and the original assignments.


## 3. Citation

In Bibtex format:

```bibtex
@misc{pyssl2023giakoumoglou,  
     author = {Nikolaos Giakoumoglou},  
     title = {PySSL: A PyTorch implementation of Self-Supervised Learning (SSL) methods},  
     year = {2023},  
     publisher = {GitHub},  
     journal = {GitHub repository},  
     howpublished = {\url{https://github.com/giakou4/pyssl}},  
   }  
```

## 4. Support
Reach out to me:
- [giakou4's email](mailto:giakou4@gmail.com "giakou4@gmail.com")
