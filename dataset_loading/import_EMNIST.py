import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import random
from dataset_loading.mappings import char_map
print("Importing data")

training_data = datasets.EMNIST(
    root="data",
    split="bymerge",
    train=True,
    download=True,
    transform=v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Lambda(lambda x: torch.where(x > 0.7, 1.0, 0.0)),
    v2.RandomRotation(degrees=(90, 90)),
    v2.RandomVerticalFlip(p=1.0)
    ])
)
training_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)
test_data = datasets.EMNIST(
    root="data",
    split="bymerge",
    train=False,
    download=True,
    transform=v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Lambda(lambda x: torch.where(x > 0.9, 1.0, 0.0)),
    v2.RandomRotation(degrees=(90, 90)),
    v2.RandomVerticalFlip(p=1.0)
    ])
)
test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)

print(len(test_data))
image, label = training_data[random.randint(0, len(test_data))]
print(chr(char_map[str(label)]))
plt.imshow(image.permute(1, 2, 0).detach().cpu(), cmap='gray')
plt.savefig('EMNIST_image.png')