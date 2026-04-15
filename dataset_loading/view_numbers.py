import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
from dataset_loading.import_data import training_data

print("Showing Numbers")
z = 0
fig, ax = plt.subplots(nrows=30, ncols=30, figsize=(60, 60))
for y, axi in enumerate(ax.flat):
    for i in range(0, 30):
        x = random.randint(0, 59999)
        image_tensor, label_tensor = training_data[x]
        axi.imshow(image_tensor.permute(1, 2, 0).detach().cpu(), cmap='gray')
        axi.axis('off')
        print(f"Image {z}/900 loaded")
        z += 1

plt.savefig("large_test")