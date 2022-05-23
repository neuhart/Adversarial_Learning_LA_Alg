import torch
import torchvision
from torch.utils.data import Subset
from easydict import EasyDict
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from absl import flags
import time
import project_utils
import torchvision.transforms as transforms

FLAGS = flags.FLAGS


def settings(dataset, nb_epochs=10, adv_train=True, fgsm_att=False, pgd_att=False):
    """defines settings via global variables"""
    flags.DEFINE_string("dataset", dataset, "Dataset to train and test on.")
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


def my_training(train_loader, net, optimizer, device):
    """trains the network on the provided training set using the given optimizer"""

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    results = []
    for epoch in range(1, FLAGS.nb_epochs + 1):
        start_t = time.time()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                x = projected_gradient_descent(net, x, 0.3, 0.01, 40, np.inf)

            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()

            if project_utils.get_optim_name(optimizer) in ['ExtraSGD', 'ExtraAdam']:
                # For Extra-SGD/Adam, we need an extrapolation step
                optimizer.extrapolation()
                optimizer.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
            elif project_utils.get_optim_name(optimizer) in ['Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']:
                # For LA-Extra Algs we need to perform an extrapolation step with the inner optimizer
                optimizer.optimizer.extrapolation()
                optimizer.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
                
            optimizer.step()
            train_loss += loss.item()
        end_t = time.time()
        print(
            "epoch: {}/{}, train loss: {:.3f} computed in {:.3f} seconds".format(
                epoch, FLAGS.nb_epochs, train_loss/len(train_loader), end_t-start_t
            )
        )
        results.append(train_loss)
    project_utils.save_results(optimizer, dataset=FLAGS.dataset, adv_train=FLAGS.adv_train, results=results)


def my_evaluation(test_loader, net, device):
    """performs model evaluation/testing"""

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


def ld_dataset(model):
    """
    Load training and test data.
    1) https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)
    """
    if model.__class__.__name__ == 'ResNet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  # convert PIL image into tensor and transform to match ResNet50 requirements (see 1))
        if FLAGS.dataset == 'MNIST':  # need to transform 1-channel MNIST images to 3-channel format for ResNet model
            transform = transforms.Compose([transforms.Grayscale(3), transform])
    elif model.__class__.__name__ == 'BasicCNN':
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        raise 'Transformation not implemented for {}'.format(model.__class__.__name__)

    trainset = getattr(torchvision.datasets, FLAGS.dataset)(root='./data', train=True,
                                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    if project_utils.yes_no_check('Run on whole dataset?') is False:
        trainset = Subset(trainset,
                          indices=range(project_utils.int_query('[0,{}] - Subset size'.format(len(trainset)))))
        # Loads only a subset (given by the indices) of the dataset

    batch_size = project_utils.int_query('Batch size')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    # load in training set: num_workers = how many subprocesses to use for data loading.

    testset = getattr(torchvision.datasets, FLAGS.dataset)(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # load in test set

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return EasyDict(train=trainloader, test=testloader, classes=classes)


