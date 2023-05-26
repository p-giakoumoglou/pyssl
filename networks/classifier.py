import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """ Linear classifier for evaluation """
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    
