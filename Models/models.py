import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(torch.nn.Module):
    """Basic CNN architecture for the MNIST dataset."""

    def __init__(self, in_channels=1):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(
            128 * 4 * 4, 128
        )  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CIFAR10_CNN(torch.nn.Module):
    """Basic CNN architecture for the CIFAR10 dataset"""

    def __init__(self, in_channels=3):  # method thats called when a new CNN instance is created
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        # in_channels = 1 = number of input stacks (e.g, colours),
        # out_channels= 64 = number of output stacks/number of filters in the layer ,
        # kernel_size= 8 = size of each convolution filter (8x8),
        # stride=1 (shift of kernel/filter from left to right, top to bottom when convolution is performed)
        # in this case: convoluted image with width W_after= [W_before-(kernel size-1)]/stride and
        # height H_after = [H_before-(kernel size-1)]/stride , else see 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, 10)  # image dimensions 3x3 with 128 channels (128,3,3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)  # reshapes tensor from (batch_size, #channels, height, width) to
        # (batch_size, 128*3*3); (image dimensions 3x3 with 128 channels (128,3,3))
        x = self.fc(x)
        return x
