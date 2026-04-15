import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt

print("Importing data")

training_data = datasets.QMNIST(
    root="data",
    what="train",
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Lambda(lambda x: torch.where(x > 0, 1.0, 0.0))])
)
training_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)
test_data = datasets.QMNIST(
    root="data",
    what="test",
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)
print(training_dataloader)
print(f"Succesfuly imported {len(training_data)} training images")