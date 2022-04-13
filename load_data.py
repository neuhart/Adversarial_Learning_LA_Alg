import torch
import torchvision
from easydict import EasyDict
import matplotlib.pyplot as plt
import numpy as np


def ld_cifar10(transform, batch_size):
    """Load training and test data."""

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    # load in training set: num_workers = how many subprocesses to use for data loading.

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # download test set, store into ./data and apply transform

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, test=testloader, classes=classes)


def imshow(dataloader, batch_size, classes, inv_transform):
    """Plot a batch of images
    dataloader: dataloader from which to get images
    classes: tuple/list of classes
    inv_transform: inversion of transformation 
    """
    data_iter = iter(dataloader)
    images, labels = data_iter.next()
    img = torchvision.utils.make_grid(images)
    img = inv_transform(img)
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose axes (0,1,2) to (1,2,0)
    # (90 degrees turn and making sure the colour values are in the third axis)
    plt.show()
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))  # print labels
    # plt.imshow(np.transpose(npimg, (0, 2, 1)))  # transpose axes (0,1,2) to (1,2,0)
    # plt.show()


import torchvision.transforms as transforms

transformation = transforms.Compose(
    [transforms.ToTensor()])  # convert PIL image into tensor

batchsize = 4  # number of samples per batch

if __name__ == '__main__':
    loaded_data = ld_cifar10(transformation, batch_size=batchsize)
    imshow(dataloader=loaded_data.train, batch_size=batchsize, classes=loaded_data.classes)
