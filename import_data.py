import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.QMNIST(
    root="data",
    what="train",
    download=True
)

print(len(training_data))