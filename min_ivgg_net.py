import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam

import torch
from torch.utils.data import  DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

train_data = CIFAR10(root='./', train=True, download=True, transform=ToTensor())
test_data = CIFAR10(root='./', train=False, download=True, transform=ToTensor())


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG16(nn.Module):
    def __init__(self, num_classes =10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        #
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer6 = nn.AdaptiveAvgPool2d(output_size=(3, 3))

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=576, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=num_classes, bias=True),
        )

    def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = out.view(out.size(0), -1)
            out = self.layer7(out)

            return out



model = VGG16(num_classes=10)



def train(train_loader, n_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = 0.01)
    total_step = len(train_loader)

    for  epoch in range(n_epoch):
        print('[LOG] we in train')
        for i, (image, label) in enumerate(train_loader):

            images, labels = image.to(device), label.to(device)
            #forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #log
        print(f'[LOG] Epoch[{epoch + 1}/{n_epoch}], Step[{i+1}/{total_step}], Loss{loss.item():.4f}')

def test(test_loader):
    correct, total = 0, 0
    with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                del images, labels, outputs
            print('Accuracy of the network on test images: {} %'.format(100 * correct / total))

#
# images = torch.randn(30, 3, 32, 32)
# labels = torch.randint(0, 10, (30,))
#
# dataset = TensorDataset(images, labels)

model = model.to(device)
train(train_loader, 10)



