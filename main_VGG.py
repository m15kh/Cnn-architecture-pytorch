import torch
from torch.utils.data import  DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor



train_data = CIFAR10(root='/data/', train=True, download=True, transform=ToTensor())
test_data = CIFAR10(root='/data/', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


