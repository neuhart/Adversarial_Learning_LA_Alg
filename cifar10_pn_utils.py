import torch
import torchvision
from torch.utils.data import Subset
from torch.utils.data import random_split
from easydict import EasyDict
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from absl import flags
import time
import project_utils

FLAGS = flags.FLAGS


def settings(nb_epochs=10, adv_train=True, fgsm_att=False, pgd_att=False):
    flags.DEFINE_integer("nb_epochs", nb_epochs, "Number of epochs.")
    flags.DEFINE_bool(
        "adv_train", adv_train, "Use adversarial training (on PGD adversarial examples)."
    )
    flags.DEFINE_bool(
        "fgsm_att", fgsm_att, "Use FGSM attack in evaluation"
    )
    flags.DEFINE_bool(
        "pgd_att", pgd_att, "Use PGD attack in evaluation"
    )


def cifar10_training(train_loader, net, optimizer, device, adv_train=False):

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    results = []
    for epoch in range(1, FLAGS.nb_epochs + 1):
        start_t = time.time()
        train_loss = 0.0
        for x, y in train_loader:  # take batches of batch_size many inputs stored in x and targets stored in y
            x, y = x.to(device), y.to(device)
            if adv_train:
                x = projected_gradient_descent(net, x, 0.3, 0.01, 40, np.inf)
            optimizer.zero_grad()  # explained in 3). Sets the gradient to zero
            loss = loss_fn(net(x), y)  # creates a new loss_fn (torch.nn.crossentropyloss) class instance
            loss.backward()  # computes the gradient - see also 4)
            if optimizer.__class__.__name__ in ['LA-ExtraAdam', 'LA-ExtraSGD', 'ExtraSGD', 'ExtraAdam']:
                # For Extra-SGD/Adam, we need an extrapolation step
                optimizer.extrapolation()
                optimizer.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
            optimizer.step()  # updates the parameters - see also 4)
            train_loss += loss.item()  # extracts loss value
        end_t = time.time()
        print(
            "epoch: {}/{}, train loss: {:.3f} computed in {:.3f} seconds".format(
                epoch, FLAGS.nb_epochs, train_loss/len(train_loader), end_t-start_t
            )
        )
        results.append(train_loss)
    project_utils.save_results(optimizer, bool_adv_train=FLAGS.adv_train, results=results)


def cifar10_evaluation(test_loader, net, device):
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        _, y_pred = net(x).max(1)  # model prediction on clean examples

        report.nb_test += y.size(0)  # counts how many examples are in the batch
        report.correct += y_pred.eq(y).sum().item()  # counts how many examples in the batch are predicted correctly

        if FLAGS.fgsm_att:
            x_fgm = fast_gradient_method(net, x, 0.3, np.inf)
            _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples

            report.correct_fgm += y_pred_fgm.eq(y).sum().item()  # counts correctly predicted fgsm examples

        if FLAGS.pgd_att:
            x_pgd = projected_gradient_descent(net, x, 0.3, 0.01, 40, np.inf)
            _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples

            report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # counts correctly predicted pgd examples

    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )

    if FLAGS.fgsm_att is True:
        print(
            "test acc on FGM adversarial examples (%): {:.3f}".format(
                report.correct_fgm / report.nb_test * 100.0
            )
        )
    if FLAGS.pgd_att is True:
        print(
            "test acc on PGD adversarial examples (%): {:.3f}".format(
                report.correct_pgd / report.nb_test * 100.0
            )
        )


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
    """Load SUBSET of cifar10 dataset for training data."""

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

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, test=testloader, classes=classes)


def dataset_spec(transform):
    """
    Specify whether the whole dataset or just a subset of it shall be loaded
    If Subset, specify size of subset
    Currently no training/validation split possible
    Currently no choice of RANDOM subset possible
    """
    if project_utils.yes_no_check('Run on whole dataset?'):
        return ld_cifar10(transform=transform, batch_size=project_utils.int_query('Batch size'))
    else:
        s_size = project_utils.int_query('Subset size')
        return ld_cifar10_subset(transform=transform, indices=range(s_size),
                                 batch_size=project_utils.int_query('Batch size'))
