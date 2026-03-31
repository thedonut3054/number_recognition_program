import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.QMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

image_tensor, label_tensor = training_data[0]
plt.imshow(image_tensor.numpy()[0],cmap='gray')
plt.savefig("test")