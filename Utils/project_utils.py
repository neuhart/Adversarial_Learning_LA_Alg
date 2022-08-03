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
        scores(pd.Dataframe): Dataframe containing the accuracy scores for each optimizer
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


def save_valid_results(settings, optimizer, scores, attack=None):
    """
    saves validation results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        scores(list): list of test accuraries computed after every epoch
        attack(str): states which attack was used for evaluation, clean=No attack
        """

    if settings.adv_train:
        if attack is not None:
            # create directories if necessary
            Path("results/{}/adv_{}_valid_results".format(settings.dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = 'results/{}/adv_{}_valid_results/{}.csv'.format(
                settings.dataset, attack, get_optim_name(optimizer))
        else:
            Path("results/{}/adv_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
            filename = 'results/{}/adv_valid_results/{}.csv'.format(settings.dataset,
                                                                    get_optim_name(optimizer))
    else:
        if attack is not None:
            Path('results/{}/clean_{}_valid_results'.format(settings.dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = 'results/{}/clean_{}_valid_results/{}.csv'.format(
                settings.dataset, attack, get_optim_name(optimizer))
        else:
            Path("results/{}/clean_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
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
                          'Lookahead-SGD', 'Lookahead-Adam', 'Lookahead-OGDA',
                          'Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']
    print('Separate by ","!', 'Type "A" for all optimizers!')
    optims_list = input('Select optimizers \n{}:'.format(implemented_optims))
    if optims_list == 'A':
        return implemented_optims
    optims_list = optims_list.split(',')  # separate by ',' and convert to list
    optims_list = list(map(lambda i: i.replace('LA-', 'Lookahead-'), optims_list))  # replaces abbreviation
    for optim in optims_list:
        assert optim in implemented_optims, '{} not implemented'.format(optim)
    return optims_list


def set_optim(settings, optim, model):
    """Sets optimizer
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optim(str): name of optimizer to use
        model(torch.nn.Module): model providing the parameters to be optimized
    """
    hyperparams = get_opt_hyperprams(settings, optim)
    if optim == 'Lookahead-SGD':
        optimizer = Lookahead.Lookahead(
            torch.optim.SGD(model.parameters(), lr=hyperparams[0]), la_steps=hyperparams[1], la_alpha=hyperparams[2])
    elif optim == 'Lookahead-Adam':
        optimizer = Lookahead.Lookahead(
            torch.optim.Adam(model.parameters(), lr=hyperparams[0]), la_steps=hyperparams[1], la_alpha=hyperparams[2])
    elif optim == 'Lookahead-ExtraSGD':
        optimizer = Lookahead.Lookahead(
            extragradient.ExtraSGD(model.parameters(), lr=hyperparams[0]), la_steps=hyperparams[1], la_alpha=hyperparams[2])
    elif optim == 'Lookahead-ExtraAdam':
        optimizer = Lookahead.Lookahead(
            extragradient.ExtraAdam(model.parameters(), lr=hyperparams[0]), la_steps=hyperparams[1], la_alpha=hyperparams[2])
    elif optim == 'Lookahead-OGDA':
        optimizer = Lookahead.Lookahead(
            OGDA.OGDA(model.parameters(), lr=hyperparams[0]), la_steps=hyperparams[1], la_alpha=hyperparams[2])
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams[0])
    elif optim == 'ExtraAdam':
        optimizer = extragradient.ExtraAdam(model.parameters(), lr=hyperparams[0])
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams[0])
    elif optim == 'OGDA':
        optimizer = OGDA.OGDA(model.parameters(), lr=hyperparams[0])
    elif optim == 'ExtraSGD':
        optimizer = extragradient.ExtraSGD(model.parameters(), lr=hyperparams[0])
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
        settings = EasyDict(nb_epochs=50, adv_train=True)
    else:
        settings = EasyDict(
            nb_epochs=query_int('Number of epochs'),
            adv_train=yes_no_check('Adversarial Training?')
        )
    return settings


def get_opt_hyperprams(settings, optim):
    """Returns optimal hyperparameters found by gridsearch
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optim(str): Name of optimizer for which hyperparams need to be fetched
    Return:
        opt_hyperparams(tuple)
    """
    if settings.adv_train:
        path = 'Hyperparam_tuning/{}/adv_pgd_test_results/{}.csv'.format(settings.dataset, optim)
    else:
        path = 'Hyperparam_tuning/{}/clean_pgd_test_results/{}.csv'.format(settings.dataset, optim)
    df = pd.read_csv(path)
    opt_hyperparams = df.idxmax(axis=1)[0]  # string containing best hyperparameter settings
    if optim.startswith('Lookahead'):
        opt_hyperparams = opt_hyperparams.split(',')  # split and convert to list
        # get hyperparameter values
        opt_hyperparams = (float(opt_hyperparams[0][3:]), int(opt_hyperparams[1][6:]), float(opt_hyperparams[2][6:]))
    else:
        opt_hyperparams = (float(opt_hyperparams[3:]), 0, 0)  # complemented with two 0's to keep the same format
    return opt_hyperparams
