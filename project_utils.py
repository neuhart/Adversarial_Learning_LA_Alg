import pandas as pd
import torch
import torchvision
from easydict import EasyDict
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import random_split


def ld_cifar10(transform, batch_size, valid_size=None):
    """Load training and test data.
    OPTIONAL: valid_size: Size of validation set
    create random_split of training data
    """

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    if valid_size is not None:  # create train/validation split of training set

        assert valid_size < len(trainset), \
            "Size of validation set has to be smaller than {}".format(len(trainset))

        trainset, validset = random_split(trainset, lengths=[len(trainset)-valid_size,valid_size]
                                          , generator=torch.Generator().manual_seed(42))

        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
        # load in training set: num_workers = how many subprocesses to use for data loading.

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    if valid_size is None:  # no validation set
        validloader = None
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, valid=validloader, test=testloader, classes=classes)


def ld_cifar10_subset(transform, indices, batch_size):
    """Load SUBSET of cifar10 dataset for training and test data."""

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    trainset = Subset(trainset, indices=indices)
    # Loads only a subset (given by the indices) of the dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    # load in training set: num_workers = how many subprocesses to use for data loading.

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # download test set, store into ./data and apply transform

    testset = Subset(testset, indices=indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, test=testloader, classes=classes)


def dataset_spec(transform,batch_size):
    """
    Specify whether the whole dataset or just a subset of it shall be loaded
    If Subset, specify size of subset
    Currently no training/validation split possible
    Currently no choice of RANDOM subset possible
    """
    if yes_no_check('Run on whole dataset?'):
        return ld_cifar10(transform=transform, batch_size=batch_size)
    else:
        s_size = int_query('Subset size')
        return ld_cifar10_subset(transform=transform, indices=range(s_size), batch_size=batch_size)


def imshow(dataloader, batch_size, classes, inv_transform=None):
    """Plot a batch of images
    dataloader: dataloader from which to get images
    classes: tuple/list of classes
    inv_transform (OPTIONAL): inversion of transformation
    """
    data_iter = iter(dataloader)
    images, labels = data_iter.next()
    img = torchvision.utils.make_grid(images)
    if inv_transform is not None:
        img = inv_transform(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose axes (0,1,2) to (1,2,0)
    # (90 degrees turn picture stack and making sure the colour values are in the third axis)
    plt.show()
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))  # print labels


def yes_no_check(question):
    """
    Ask a 'yes or no'-question
    return True for yes and False for no
    """
    x = input(question + ' (y/n):')
    while x not in ['y', 'n']:
        print('expected "y" or "n" ')
        x = input(question + ' (y/n):')
    return True if x == 'y' else False


def int_query(query):
    """
    Query an integer
    """
    x = input(query + ' (int):')
    while isinstance(x, int) is False:
        try:
            x = int(x)
        except:
            print('Integer expected')
            x = input(query + ' (int):')
    return x


def save_results(optimizer, results):
    # assert optimizer in ['Adam', 'Lookahead', 'SGD']
    filename = 'data/results.csv'
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()
    df = pd.concat([df, pd.Series(results, name=optimizer)], axis=1)
    df.to_csv(filename, index=False)


def get_opt():
    x = input('Lookahead or Adam? Choice:')
    assert x in ['Lookahead', 'Adam'], 'Choose between "Lookahead" or "Adam"'
    return x