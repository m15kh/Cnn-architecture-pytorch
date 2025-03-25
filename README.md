# CNN Architecture Implementation in PyTorch

This repository contains implementations of CNN architectures in PyTorch, with a focus on VGG16 and its variants.

## Project Structure

- `Cnn-architecture-pytorch/`
  - `main_VGG.py` - Data loading utilities for CIFAR10
  - `min_ivgg_net.py` - A minimal implementation of VGG network
  - `VGGNet.py` - Complete VGG16 implementation with training utilities
  - `vgg16.ipynb` - Jupyter notebook exploring VGG16 architecture

## Features

- VGG16 network implementation from scratch
- Training and testing utilities
- CIFAR10 dataset integration
- GPU support with automatic device detection

## Usage

### Training a VGG Model

```python
from Cnn-architecture-pytorch.min_ivgg_net import VGG16, train, test
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# Load data
train_data = CIFAR10(root='./', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Create model
model = VGG16(num_classes=10)

# Train model
train(train_loader, n_epochs=10)
```

### Testing the Model

```python
from Cnn-architecture-pytorch.min_ivgg_net import VGG16, test
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# Load test data
test_data = CIFAR10(root='./', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Test model
test(test_loader)
```

## Requirements

- PyTorch >= 1.0.0
- torchvision >= 0.2.0
- Python >= 3.6

## Implementation Details

The repository includes multiple VGG16 implementations:

1. **Minimal Implementation**: A simplified version with fewer layers for faster training
2. **Full VGG16**: Complete implementation following the original architecture from the paper

Both implementations are adapted for the CIFAR10 dataset which contains 10 classes of 32x32 RGB images.

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - Original VGG paper by Simonyan and Zisserman
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
