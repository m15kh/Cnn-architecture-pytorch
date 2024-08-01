import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import SGD

device = ('cuda' if torch.cuda.is_available() else 'cpu')
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
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1),
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
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer15 = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.layer16 = nn.Sequential(
            nn.Linear(in_features=30*25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1000, out_features=num_classes, bias=True),
        )

    def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = self.layer8(out)
            out = self.layer9(out)
            out = self.layer10(out)
            out = self.layer11(out)
            out = self.layer12(out)
            out = self.layer13(out)
            out = self.layer14(out)
            out = self.layer15(out)
            out = out.view(out.size(0), -1)
            out = self.layer16(out)

            return out



model = VGG16(num_classes=10)

# Create a random tensor with shape (30, 3, 28, 28)
input_tensor = torch.randn(30, 3, 280, 280)

# Pass the tensor through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)

def train(train_loader, n_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.01, weight_decay = 0.005, momentum = 0.9)
    total_step = len(train_loader)

    for  epoch in range(n_epoch):
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
            print(f'Epoch[{epoch + 1}/{n_epoch}], Step[{i+1}/{total_step}], Loss{loss.item():.4f}')

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


# num_classes = 10
# model = VGG16(num_classes).to(device)
# testtorch = torch.randn(4, 3, 28, 28)
# train(testtorch, 1)