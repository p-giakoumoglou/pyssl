# Self-Supervised Learning

Self-Supervised Learning (SSL) is a subfield of machine learning where the model learns to predict part of the data from other parts of the same data. This is often used in an unsupervised manner where the labels used for learning are generated from the data itself. This approach helps in creating representations of the data that can be used for downstream tasks such as classification, detection, and segmentation. Self-Supervised Learning has been successful in a variety of domains such as computer vision, natural language processing, and reinforcement learning.

## 1 SimCLR

[SimCLR](https://arxiv.org/abs/2002.05709) (Simple Contrastive Learning of Visual Representations) is a framework for self-supervised learning of visual representations. It aims to learn representations by maximizing agreement between differently augmented views of the same data sample. 

## 2 SupCon

[SupCon](https://arxiv.org/abs/2004.11362) (Supervised Contrastive Learning) is another self-supervised learning method. While SimCLR uses a single positive pair (two augmented views of the same image) for each image in the batch, SupCon allows multiple positive pairs for each image. The key idea of SupCon is to use the labels available in supervised learning to define the positive and negative samples, which makes this method a hybrid of supervised and self-supervised learning.

## 3 SimSiam

[SimSiam](https://arxiv.org/abs/2011.10566) (Simple Siamese Networks) is a self-supervised learning approach that uses a simple framework to learn visual representations. The method involves two identical networks (hence the term "Siamese") that generate representations for two augmented views of the same image. The objective is to make these two representations as similar as possible. Unlike SimCLR and SupCon, SimSiam does not use negative pairs in its learning objective. This eliminates the necessity of large batch sizes and makes the method more computationally efficient.

## 4 BYOL (TBD)

[BYOL](https://arxiv.org/abs/2006.07733) (Bootstrap Your Own Latent) is a novel self-supervised learning method that, like SimSiam, does not rely on negative pairs for training. Instead, it learns representations by comparing the representations of two augmented views of the same image. BYOL introduces a concept of a "target network" which is a moving average of the main network. The main network learns to align its representations with the slowly moving target network, which provides a form of consistency regularization.

## 5 SWAV (TBD)

[SWAV](https://arxiv.org/abs/2006.09882) (Swapping Assignments between Views) is a self-supervised learning method that introduces a new way of assigning labels for the contrastive loss function. It uses a clustering algorithm to group similar representations together and assign them the same pseudo-label. SWAV introduces a unique "swap" operation where the assignments of the pseudo-labels are swapped between different views of the same image. This forces the model to learn consistent representations across different augmentations.

## 6 MoCo (TBD)

MoCo (Momentum Contrast) maintains a dynamic dictionary of data samples in the memory for generating negative samples, providing a large number of negative samples for contrastive learning.

## 7 InfoNCE (TBD)

InfoNCE (Information Noise-Contrastive Estimation) formulates contrastive learning as noise-contrastive estimation in the information theory framework.
