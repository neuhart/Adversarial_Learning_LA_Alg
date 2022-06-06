import torch
import torchvision
from torch.utils.data import Subset
from easydict import EasyDict
from absl import flags
import project_utils


def ld_dataset(dataset_name, transform):
    """
    Load training and test data.
    Arguments:
        dataset_name(str): name of the dataset
        transform(torchvision.transforms): input transformation
    returns:
    EasyDict: easydict dictionary containing:
        trainloader: torch dataloader for train data
        testloader: torch dataloader for test data
        classes: tuple of class names
    """

    trainset = getattr(torchvision.datasets, dataset_name)(root='./data', train=True,
                                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    if project_utils.yes_no_check('Run on whole dataset?') is False:
        trainset = Subset(trainset,
                          indices=range(project_utils.query_int('[0,{}] - Subset size'.format(len(trainset)))))
        # Loads only a subset (given by the indices) of the dataset

    batch_size = project_utils.query_int('Batch size')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    # load in training set: num_workers = how many subprocesses to use for data loading.

    testset = getattr(torchvision.datasets, dataset_name)(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, test=testloader, classes=classes)


