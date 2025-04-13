import numpy as np
import torch
from torch import nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10

    # your code here

    # return model instance (None is just a placeholder)

    return nn.Sequential(
      nn.Linear(784, 256),
      nn.ReLU(),
      nn.Linear(256, 16),
      nn.ReLU(),
      nn.Linear(16, 10),
    )

def count_parameters(model):
    # your code here
    size = 0
    for param in model.parameters():
      size += param.numel()
    # верните количество параметров модели model
    return size
