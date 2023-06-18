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

## 2. Methods

The SSL method implementation was based on [lucidrains](https://github.com/lucidrains) implementation of [BYOL](https://github.com/lucidrains/byol-pytorch).

For the following section, assume a backbone, e.g., a ResNet-50, and an input image of size 224 in a batch of 4, i.e.,

```python
import torchvision

backbone = torchvision.models.resnet50(pretrained=False)
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()

x = torch.rand(4, 3, 224, 224)
```

### 2.1 Barlow Twins

Barlow Twins is a SSL method that aims to learn representations by making the outputs of a neural network to be as similar as possible for two augmented views of the same image, while reducing the redundancy between the output components.

```python
model = BarlowTwins(backbone, feature_size, projection_dim=8192, hidden_dim=8192, lamda=0.005)
```

### 2.2 BYOL

BYOL (Bootstrap Your Own Latent) is an SSL method that learns representations by predicting one view of an input from another view of the same input, without the necessity for negative pairs which is the case for most contrastive learning methods.

```python
model = BYOL(backbone, feature_size, projection_dim=256, hidden_dim=4096, tau=0.996)
```

### 2.3 DINO

DINO (Distillation of Self-supervised Networks) is a method that trains a student network using the outputs of a teacher network, where the teacher network is an exponentially moving average of the student network. The key idea is to use distillation loss to capture information from different viewpoints of the same image.


```python
model = DINO(backbone, feature_size, projection_dim=256, hidden_dim=2048, bottleneck_dim=256, temp_s=0.1, temp_t=0.5, m=0.5, lamda=0.996, num_crops=6)
```

### 2.4 MOCO, MoCov2, MoCov3

MOCO (Momentum Contrast) series are methods that utilize a dynamic dictionary implemented with a queue and a moving-averaged encoder. The methods aim to maximize similarity between a query and its positive key and minimize similarity between the query and negative keys (v2 and v3 are updated versions with improvements over the original).

```python
model = MoCo(backbone, feature_size, projection_dim=128, K=65536, m=0.999, temperature=0.07)
model = MoCoV2(backbone, feature_size, projection_dim=128, K=65536, m=0.999, temperature=0.07)
model = MoCoV3(backbone, feature_size, projection_dim=256, hidden_dim=2048, temperature=0.5, m=0.999)
```

### 2.5 SimCLR, SimCLRv2

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a method that utilizes a contrastive loss function to distinguish between similar and dissimilar images. The v2 version introduces a simple method for augmenting the data sample with a learnable nonlinear transformation.

```python
model = SimCLR(backbone, feature_size, projection_dim=128, temperature=0.5)
model = SimCLRv2(backbone, feature_size, projection_dim=128, temperature=0.5)
```

### 2.6 SimSiam

SimSiam (Simple Siamese) aims to learn representations by predicting one view of an input from another view of the same input, similar to BYOL. However, unlike BYOL, SimSiam does not use batch normalization in the prediction MLP, and does not require the use of a momentum encoder or moving average.

```python
model = SimSiam(backbone, feature_size, projection_dim=2048, hidden_dim_proj=2048, hidden_dim_pred=512)
```

### 2.7 SupCon

SupCon (Supervised Contrastive Learning) is a method that uses a contrastive loss function but also makes use of labels, if available, to further refine the learning of representations by encouraging dissimilarities between representations of different classes.

```python
model = SupCon(backbone, feature_size, projection_dim=128, temperature=0.07)
```

### 2.8 SwAV

SwAV (Swapping Assignments between multiple Views of the same image) is a clustering-based method which learns representations by swapping the cluster assignments between different views of the same image, and then minimizing the difference between the swapped assignments and the original assignments.

```python
model = SwAV(backbone, feature_size, projection_dim=128, hidden_dim=2048, temperature=0.1, epsilon=0.05, sinkhorn_iterations=3, num_prototypes=3000, queue_length=64, use_the_queue=True, num_crops=6)
```

## 3. Training

The models can directly output the loss, i.e., ```loss = model(x)``` or ```loss = model.forward(x)``` so as to integrate smoothly with the training loop (see main.py).

```python
import torch
import torchvision

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize backbone (resnet50)
backbone = torchvision.models.resnet50(pretrained=False)
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()

# initialize ssl method
model = builders.SimCLR(backbone, feature_size, image_size=32)
model = model.to(device)
    
# load fake CIFAR-like dataset
dataset = torchvision.datasets.FakeData(2000, (3, 32, 32), 10, torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# switch to train mode
model.train()

# epoch training
for epoch in range(10):
    for i, (images, _) in enumerate(loader):
        images = images.to(device)

        # zero the parameter gradients
        model.zero_grad()

        # compute loss
        loss = model(images)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
```

Note, that each method uses a default augmentation for training, according to original implementation, and the model requires the image size, mean, and standard deviation of the dataset. 

For example:

```python
kwargs = {
    'image_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

model = BYOL(backbone, feature_size, projection_dim=8192, hidden_dim=8192, lamda=0.005, **kwargs)

```

## 4. Inference

For inference, use either ```model.encoder(x)``` to get the projection vector (backbone and projector's representations), or use ```model.backbone(x)``` to get the feature vector (backbone representations). The model automatically assigns the ```encoder``` to the corresponding encoder of each method (e.g., student network for BYOL or queue encoder for MoCo series).

## 5. Citation

In Bibtex format:

```bibtex
@misc{pyssl2023giakoumoglou,
   author = {Nikolaos Giakoumoglou and Paschalis Giakoumoglou},
   title = {PySSL: A PyTorch implementation of Self-Supervised Learning (SSL) methods},
   year = {2023},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/giakou4/pyssl}},
}
```

## 5. Support
Reach out to me:
- [giakou4's email](mailto:giakou4@gmail.com "giakou4@gmail.com")
