from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from easydict import EasyDict
from Optimizer import Lookahead, extragradient, OGDA


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


def save_train_results(settings, optimizer, results):
    """
    saves results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        results(list): list of train losses computed after every epoch
        """
    # create directories if necessary
    Path("results/{}/adv_train_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
    Path("results/{}/clean_train_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)

    filename = 'results/{}/adv_train_results/{}.csv'.format(settings.dataset, get_optim_name(optimizer)) if \
        settings.adv_train else 'results/{}/clean_train_results/{}.csv'.format(settings.dataset, get_optim_name(optimizer))
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(results, name="Column {}".format(len(df.columns)+1))], axis=1)
    df.to_csv(filename, index=False)


def save_test_results(settings, scores, attack=None):
    """
    saves results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        scores(pd.Dataframe): Dataframe containing the accuracy scores for each
        optimizer if a pgd attack has been executed, else None
        attack(str): states which attack was used for evaluation, clean=No attack

    """
    if settings.adv_train:
        if attack is not None:
            filename = 'results/{}/adv_{}_test_results.csv'.format(settings.dataset, attack)
        else:
            filename = 'results/{}/adv_test_results.csv'.format(settings.dataset)
    else:
        if attack is not None:
            filename = 'results/{}/clean_{}_test_results.csv'.format(settings.dataset, attack)
        else:
            filename = 'results/{}/clean_test_results.csv'.format(settings.dataset)

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, scores], axis=0)
    df.to_csv(filename, index=False)


def save_valid_results(settings, optimizer, scores, attack):
    """
    saves validation results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        scores(list): list of test accuraries computed after every epoch
        attack(str): states which attack was used for evaluation, clean=No attack
        """
    # create directories if necessary
    Path("results/{}/adv_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
    Path("results/{}/clean_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)

    if settings.adv_train:
        if attack is not None:
            filename = 'results/{}/adv_{}_valid_results/{}.csv'.format(
                settings.dataset, attack, get_optim_name(optimizer))
        else:
            filename = 'results/{}/adv_valid_results/{}.csv'.format(settings.dataset,
                                                                    get_optim_name(optimizer))
    else:
        if attack is not None:
            filename = 'results/{}/clean_{}_valid_results/{}.csv'.format(
                settings.dataset, attack, get_optim_name(optimizer))
        else:
            filename = 'results/{}/clean_valid_results/{}.csv'.format(
                settings.dataset, get_optim_name(optimizer))
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(scores, name="Column {}".format(len(df.columns) + 1))], axis=1)
    df.to_csv(filename, index=False)


def query_dataset():
    """queries dataset"""
    implemented_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
    x = input('Select a dataset (1-3) {}:'.format(implemented_datasets))
    assert x in implemented_datasets+['1', '2', '3'], '{} not implemented'.format(x)
    if x == '1':
        return 'MNIST'
    elif x == '2':
        return 'FashionMNIST'
    elif x == '3':
        return 'CIFAR10'
    else:
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
