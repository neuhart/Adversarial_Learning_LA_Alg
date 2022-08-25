import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import data_transformations


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


def set_model_and_transform(settings):
    """
    Instantiates correct model + data transform
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
    Returns:
        settings(EasyDict)
    """
    if settings.dataset == 'MNIST':
        settings.model = MNIST_CNN()
        settings.transform=data_transformations.standard_transform()
    elif settings.dataset == 'FashionMNIST':
        settings.transform = data_transformations.standard_transform()
        settings.model = MNIST_CNN()
    elif settings.dataset == 'CIFAR10':
        settings.transform = data_transformations.resnet_transform()
        settings.model = torchvision.models.resnet18()

    return settings
