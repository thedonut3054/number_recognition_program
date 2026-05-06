import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as v2
from Network.Network_definition import model
from dataset_loading.import_EMNIST import training_dataloader
from dataset_loading.import_EMNIST import test_dataloader
from Network.test import test_loop
from Network.training import train_loop

learning_rate = 1e-2
epochs = 50

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(epochs)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataloader, model, loss_fn, optimizer)
    correct, test_loss = test_loop(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Done Training")
print(f"Final Accuracy: {100*correct}")
torch.save(model, "Saved_Model")