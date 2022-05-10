import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import numpy as np


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


def save_results(optimizer, dataset, adv_train, results):
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


def query_dataset():
    """queries dataset to use for both training and testing"""
    implemented_datasets = ['MNIST', 'CIFAR10']
    x = input('Select a dataset {}:'.format(implemented_datasets)).upper()
    assert x in implemented_datasets, '{} not implemented'.format(x)
    return x


def query_optim():
    """queries optimizer"""
    implemented_optims = ['LA-SGD', 'LA-Adam', 'LA-ExtraAdam', 'LA-ExtraSGD', 'LA-OGDA', 'OGDA', 'SGD', 'Adam', 'ExtraSGD', 'ExtraAdam']
    x = input('Select an opimizer {}:'.format(implemented_optims))
    assert x in implemented_optims, '{} not implemented'.format(x)
    return x


def get_optim_name(optimizer):
    """returns name of the specified optimization algorithm"""
    optimizer_name = optimizer.__class__.__name__
    if optimizer_name == 'Lookahead':
        optimizer_name += '-' + optimizer.optimizer.__class__.__name__
    return optimizer_name
