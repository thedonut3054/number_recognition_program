import torch
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.io as io
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import InterpolationMode

image = io.read_image("/workspaces/number_recognition_program/feed_image/image_to_scan/number.jpg")
original_size = (216, 216)
pixel_size = 28

transform = v2.Compose([
    v2.Resize(pixel_size, interpolation=InterpolationMode.NEAREST),
    v2.Grayscale(1),
    v2.Lambda(lambda x: torch.where(x > 0, 1.0, 0.0)),
    v2.RandomInvert(1),
])
print(transform)
image = transform(image)
print(image.size)
plt.imshow(image.permute(1, 2, 0).detach().cpu(), cmap='gray')
plt.savefig('compressed image')