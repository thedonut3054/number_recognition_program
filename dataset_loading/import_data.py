import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

print("Importing data")

training_data = datasets.QMNIST(
    root="data",
    what="train",
    download=True
)

print(f"Succesfuly imported {len(training_data)} training images")