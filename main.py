# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

import builders


def main():
    
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
    dataset = datasets.FakeData(2000, (3, 32, 32), 10, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
            print('[Epoch %2d, Batch %2d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    main()
