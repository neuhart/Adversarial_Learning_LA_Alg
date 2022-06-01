import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from Optimizer import Lookahead, extragradient, OGDA
import torch


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
    query: str - question to be asked
    """
    x = input(query + ' (int):')
    while isinstance(x, int) is False:
        try:
            x = int(x)
        except:
            print('Integer expected')
            x = input(query + ' (int):')
    return x


def save_train_results(optimizer, dataset, adv_train, results):
    """
    saves results to csv-file
    optimizer: torch optimizer
    dataset: string
    adv_train: bool
    results: list
    """
    filename = 'data/{}-adv_results.csv'.format(dataset) if adv_train else 'data/{}-clean_results.csv'.format(dataset)
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(results, name=get_optim_name(optimizer))], axis=1)
    df.to_csv(filename, index=False)


def save_test_results(optimizer, dataset, adv_train, results):
    """
    saves results to csv-file
    optimizer: torch optimizer
    dataset: string
    adv_train: bool
    results: list
    """

    if adv_train:
        filename = 'data/{}-adv_test_results.csv'.format(dataset)
    else:
        filename = 'data/{}-clean_test_results.csv'.format(dataset)

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(results, name=get_optim_name(optimizer))], axis=1)
    df.to_csv(filename, index=False)


def query_dataset():
    """queries dataset to use for both training and testing"""
    implemented_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
    x = input('Select a dataset {}:'.format(implemented_datasets))
    assert x in implemented_datasets, '{} not implemented'.format(x)
    return x


def get_optims():
    """queries optimizers"""
    implemented_optims = ['SGD', 'Adam', 'OGDA', 'ExtraAdam', 'LA-SGD', 'LA-Adam', 'LA-OGDA', 'LA-ExtraAdam']
    print('Separate with ","!', 'Type "A" for all optimizers!')
    optims_list = input('Select optimizers \n{}:'.format(implemented_optims))
    if optims_list == 'A':
        return implemented_optims
    optims_list = optims_list.split(',')  # separate by ',' and convert to list
    for optim in optims_list:
        assert optim in implemented_optims, '{} not implemented'.format(optim)
    return optims_list


def set_optim(optim, model):
    """Sets optimizer
    Arguments:
        optim(str): name of optimizer to use
        model(torch.nn.Module): model providing the parameters to be optimized
    """
    if optim == 'LA-SGD':
        optimizer = Lookahead.Lookahead(torch.optim.SGD(model.parameters(), lr=1e-3))
    elif optim == 'LA-Adam':
        optimizer = Lookahead.Lookahead(torch.optim.Adam(model.parameters(), lr=1e-3))
    elif optim == 'LA-ExtraAdam':
        optimizer = Lookahead.Lookahead(extragradient.ExtraAdam(model.parameters(), lr=1e-3))
    elif optim == 'LA-OGDA':
        optimizer = Lookahead.Lookahead(OGDA.OGDA(model.parameters(), lr=1e-3))
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optim == 'ExtraAdam':
        optimizer = extragradient.ExtraAdam(model.parameters(), lr=1e-3)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    elif optim == 'OGDA':
        optimizer = OGDA.OGDA(model.parameters(), lr=1e-3)
    else:
        raise 'Wrong optimizer'

    return optimizer


def get_optim_name(optimizer):
    """returns name of the specified optimization algorithm"""
    optimizer_name = optimizer.__class__.__name__
    if optimizer_name == 'Lookahead':
        optimizer_name += '-' + optimizer.optimizer.__class__.__name__
    return optimizer_name
