import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ВАШ КОД ЗДЕСЬ
        # определите слои сети

        self.conv1 = nn.Conv2d(3, 3, (5, 5))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(3, 5, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(180, 100)
        self.fc2 = nn.Linear(100, 10)



    def forward(self, x):
        # размерность х ~ [64, 3, 32, 32]

        # ВАШ КОД ЗДЕСЬ
        # реализуйте forward pass сети

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def create_model():
    return ConvNet()
