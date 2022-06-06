import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from Optimizer import Lookahead, extragradient, OGDA
import torch
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict
from pathlib import Path


def imshow(dataloader, batch_size, classes, inv_transform=None):
    """Plot a batch of images
    dataloader(torch dataloader): dataloader from which to get images
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


def query_int(query):
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
    Arguments:
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        dataset(str): name of the dataset the model was trained on
        adv_train(bool): True for adversarial training
        results(list): list of train losses computed after every epoch
        """
    # create directories if necessary
    Path("data/results/{}/adv_results".format(dataset)).mkdir(parents=True, exist_ok=True)
    Path("data/results/{}/clean_results".format(dataset)).mkdir(parents=True, exist_ok=True)
    
    filename = 'data/results/{}/adv_results/{}-.csv'.format(dataset, get_optim_name(optimizer)) if \
        adv_train else 'data/results/{}/clean_results/{}-.csv'.format(dataset, get_optim_name(optimizer))
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(results)], axis=1)
    df.to_csv(filename, index=False)


def save_test_results(dataset, adv_train, test_results):
    """
    saves results to csv-file
    Arguments:
        dataset(str): name of the dataset the model was trained on
        adv_train(bool): True if adversarial training has been performed
        test_results(pd.Dataframe): Dataframe containing the test accuracies
    """

    if adv_train:
        filename = 'data/results/{}-adv_test_results.csv'.format(dataset)
    else:
        filename = 'data/results/{}-clean_test_results.csv'.format(dataset)

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, test_results], axis=0)
    df.to_csv(filename, index=False)


def query_dataset():
    """queries dataset to use for both training and testing"""
    implemented_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
    x = input('Select a dataset {}:'.format(implemented_datasets))
    assert x in implemented_datasets, '{} not implemented'.format(x)
    return x


def get_optims():
    """queries optimizers
    returns: optims_list (list): list of optimizers to be used"""

    implemented_optims = ['SGD', 'Adam', 'OGDA', 'ExtraAdam', 'ExtraSGD',
                          'LA-SGD', 'LA-Adam', 'LA-OGDA', 'LA-ExtraSGD', 'LA-ExtraAdam']
    print('Separate by ","!', 'Type "A" for all optimizers!')
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
    elif optim == 'LA-ExtraSGD':
        optimizer = Lookahead.Lookahead(extragradient.ExtraSGD(model.parameters(), lr=1e-5))
    elif optim == 'LA-ExtraAdam':
        optimizer = Lookahead.Lookahead(extragradient.ExtraAdam(model.parameters(), lr=1e-3))
    elif optim == 'LA-OGDA':
        optimizer = Lookahead.Lookahead(OGDA.OGDA(model.parameters(), lr=1e-4))
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optim == 'ExtraAdam':
        optimizer = extragradient.ExtraAdam(model.parameters(), lr=1e-3)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    elif optim == 'OGDA':
        optimizer = OGDA.OGDA(model.parameters(), lr=1e-4)
    elif optim == 'ExtraSGD':
        optimizer = extragradient.ExtraSGD(model.parameters(), lr=1e-5)
    else:
        raise 'Wrong optimizer'

    return optimizer


def set_lr_scheduler(optimizer):
    """Sets lr_scheduler
    Arguments:
        optimizer(torch.optim): name of optimizer to use
    """
    optim = get_optim_name(optimizer)
    if optim == 'Lookahead-SGD':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'Lookahead-Adam':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'Lookahead-ExtraSGD':
        scheduler = MultiStepLR(optimizer, milestones=[13, 25], gamma=0.1)
    elif optim == 'Lookahead-ExtraAdam':
        scheduler = MultiStepLR(optimizer, milestones=[13, 25], gamma=0.1)
    elif optim == 'Lookahead-OGDA':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'Adam':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'ExtraAdam':
        scheduler = MultiStepLR(optimizer, milestones=[13, 25], gamma=0.1)
    elif optim == 'SGD':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'OGDA':
        scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    elif optim == 'ExtraSGD':
        scheduler = MultiStepLR(optimizer, milestones=[13, 25], gamma=0.1)
    else:
        raise 'Wrong optimizer'

    return scheduler


def get_optim_name(optimizer):
    """returns name of the specified optimization algorithm"""
    optimizer_name = optimizer.__class__.__name__
    if optimizer_name == 'Lookahead':
        optimizer_name += '-' + optimizer.optimizer.__class__.__name__
    return optimizer_name


def g_settings():
    """queries hyperparameters and general settings
    returns: settings (EasyDict): dict with settings"""

    if yes_no_check('Run on standard settings?'):
        settings = EasyDict(nb_epochs=50, adv_train=True, fgsm_att=False, pgd_att=False)
    else:
        settings = EasyDict(
            nb_epochs=query_int('Number of epochs'),
            adv_train=yes_no_check('Adversarial Training?'),
            fgsm_att=yes_no_check('FGSM Attack during testing?'),
            pgd_att=yes_no_check('PGD Attack during testing')
        )
    return settings
