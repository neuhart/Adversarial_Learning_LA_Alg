import pandas as pd
import torchvision.models
from Utils import data_utils, project_utils
from Models import models, data_transformations
from Optimizer.Lookahead import Lookahead
from Optimizer.OGDA import OGDA
from Optimizer.extragradient import ExtraSGD, ExtraAdam
from easydict import EasyDict
import torch
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import time
from pathlib import Path
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


def main():
    settings = EasyDict(nb_epochs=project_utils.query_int('Number of epochs?'),
                        adv_train=project_utils.yes_no_check('Adversarial Training?'),
                        dataset=project_utils.query_dataset(),
                        fgsm_att=project_utils.yes_no_check('FGSM Attack?'),
                        pgd_att=project_utils.yes_no_check('PGD Attack?'))  # specify general settings
    settings.device = torch.device(project_utils.query_int('Select GPU [0,3]:')) if \
        torch.cuda.is_available() else torch.device('cpu')

    transform = data_transformations.resnet_transform() if settings.dataset == 'CIFAR10' else data_transformations.standard_transform()
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=transform)

    optim_list = project_utils.get_optims()

    for optim in optim_list:
        scores = pd.DataFrame()
        fgsm_scores = pd.DataFrame()
        pgd_scores = pd.DataFrame()

        for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
            settings.lr = lr
            if optim.startswith('LA-'):  # check if Lookahead is used
                settings.LA = True
                inner_optim = optim[3:]  # delete 'LA-' prefix

                for la_steps in [5, 10]:
                    settings.la_steps = la_steps
                    for la_alpha in [0.5, 0.75, 0.9]:
                        settings.la_alpha = la_alpha

                        # select correct neural network
                        net = torchvision.models.resnet18() if settings.dataset == 'CIFAR10' else models.MNIST_CNN()
                        net.to(settings.device)  # transfers to gpu if available

                        # Determine which optimizer to use
                        optimizer = Lookahead(
                            set_inner_optim(inner_optim, lr=lr, model=net), la_steps=la_steps, la_alpha=la_alpha)

                        # Train model
                        net.train()
                        train(settings, data, net, optimizer)

                        # Evaluation
                        net.eval()
                        results = evaluation(settings, data.test, net)

                        scores = pd.concat(
                            [scores, pd.Series(results.clean, name='lr={},steps={},alpha={}'.format(
                                lr, la_steps, la_alpha))], axis=1)
                        if settings.fgsm_att:
                            fgsm_scores = pd.concat(
                                [fgsm_scores, pd.Series(results.fgsm, name='lr={},steps={},alpha={}'.format(
                                    lr, la_steps, la_alpha))], axis=1)
                        if settings.pgd_att:
                            pgd_scores = pd.concat(
                                [pgd_scores, pd.Series(results.pgd, name='lr={},steps={},alpha={}'.format(
                                    lr, la_steps, la_alpha))], axis=1)
            else:
                settings.LA = False

                net = torchvision.models.resnet18() if settings.dataset == 'CIFAR10' else models.MNIST_CNN()
                net.to(settings.device)  # transfers to gpu if available

                # Determine which optimizer to use
                optimizer = set_inner_optim(optim, lr=lr, model=net)

                # Train model
                net.train()
                train(settings, data, net, optimizer)

                # Evaluation
                net.eval()
                results = evaluation(settings, data.test, net)

                scores = pd.concat(
                    [scores, pd.Series(results.clean, name='lr={}'.format(
                        lr))], axis=1)
                if settings.fgsm_att:
                    fgsm_scores = pd.concat(
                        [fgsm_scores, pd.Series(results.fgsm, name='lr={}'.format(
                            lr))], axis=1)
                if settings.pgd_att:
                    pgd_scores = pd.concat(
                        [pgd_scores, pd.Series(results.pgd, name='lr={}'.format(
                            lr))], axis=1)

        save_test_results(settings, optimizer, scores)
        if settings.fgsm_att:
            save_test_results(settings, optimizer, fgsm_scores, attack='fgsm')
        if settings.pgd_att:
            save_test_results(settings, optimizer, pgd_scores, attack='pgd')


def set_inner_optim(optim, lr, model):
    """Instantiates  (inner) optimizer
    Arguments:
        optim(str): name of optimizer to be used (Lookahead abbreviated to LA)
        lr(float): learning rate
        model(torch.nn.Module): neural network
    Returns:
        optimizer(torch.optim.Optimizer): optimizer used to train the model
    """
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'ExtraSGD':
        optimizer = ExtraSGD(model.parameters(), lr=lr)
    elif optim == 'ExtraAdam':
        optimizer = ExtraAdam(model.parameters(), lr=lr)
    elif optim == 'OGDA':
        optimizer = OGDA(model.parameters(), lr=lr)
    else:
        raise 'Wrong optimizer'
    return optimizer


def train(settings, data, model, optimizer):
    """trains the network on the provided training set using the given optimizer
    Arguments:
        settings(EasyDict): easydict dictionary containing the training settings
        data(EasyDict): easydict dictionary containing the train and test(=validation) Dataloader used for Mini-batching
        model(torch.nn.Module): model to be trained
        optimizer(torch.optim.Optimizer): optimizer used to train the model
    """

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    train_results = []
    clean_valid_results = []
    fgsm_valid_results = []
    pgd_valid_results = []

    for epoch in range(1, settings.nb_epochs + 1):
        start_t = time.time()
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(settings.device), y.to(settings.device)
            if settings.adv_train:
                x = projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf)
                x = x.detach()

            if project_utils.get_optim_name(optimizer) in ['ExtraAdam', 'ExtraSGD']:
                # For Extra-SGD/Adam, we need an extrapolation step
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.extrapolation()

            elif project_utils.get_optim_name(optimizer) in ['Lookahead-ExtraAdam', 'Lookahead-ExtraSGD']:
                # For LA-Extra Algs we need to perform an extrapolation step with the inner optimizer
                optimizer.optimizer.extrapolation()
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        end_t = time.time()
        print(
            "epoch: {}/{}, train loss: {:.3f} computed in {:.3f} seconds".format(
                epoch, settings.nb_epochs, train_loss/len(data.train), end_t-start_t
            )
        )
        train_results.append(train_loss)

        # Validation
        model.eval()
        valid_results = evaluation(settings, data.test, model)
        clean_valid_results.append(valid_results.clean)
        fgsm_valid_results.append(valid_results.fgsm)
        pgd_valid_results.append(valid_results.pgd)
        model.train()

    save_train_results(settings, optimizer, results=train_results)
    save_valid_results(settings, optimizer, scores=clean_valid_results)
    if settings.fgsm_att:
        save_valid_results(settings, optimizer, fgsm_valid_results, attack='fgsm')
    if settings.pgd_att:
        save_valid_results(settings, optimizer, pgd_valid_results, attack='pgd')


def evaluation(settings, test_loader, model):
    """performs model evaluation/testing
    Arguments:
        settings(EasyDict): easydict dictionary containing the evaluation settings
        test_loader(torch Dataloader): Dataloader used for Mini-batching
        model(torch.nn.Module): trained model to be evaluated
    returns:
    results(EasyDict): easydict dictionary containing the test accuracies
    """

    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_loader:
        x, y = x.to(settings.device), y.to(settings.device)

        _, y_pred = model(x).max(1)  # model prediction on clean examples

        report.nb_test += y.size(0)  # counts how many examples are in the batch
        report.correct += y_pred.eq(y).sum().item()  # counts how many examples in the batch are predicted correctly

        if settings.fgsm_att:
            x_fgm = fast_gradient_method(model, x, 0.3, np.inf)
            _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples

            report.correct_fgm += y_pred_fgm.eq(y).sum().item()  # counts correctly predicted fgsm examples

        if settings.pgd_att:
            x_pgd = projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf)
            _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples

            report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # counts correctly predicted pgd examples

    results = EasyDict()

    results.clean = report.correct / report.nb_test
    results.fgsm = report.correct_fgm / report.nb_test
    results.pgd = report.correct_pgd / report.nb_test

    return results


def save_train_results(settings, optimizer, results):
    """
    saves results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        results(list): list of train losses computed after every epoch
        """

    if settings.adv_train:
        # create directory if necessary
        Path("Hyperparam_tuning/{}/adv_train_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)

        filename = 'Hyperparam_tuning/{}/adv_train_results/{}.csv'.format(
        settings.dataset, project_utils.get_optim_name(optimizer))
    else:
        # create directory if necessary
        Path("Hyperparam_tuning/{}/clean_train_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)

        filename= 'Hyperparam_tuning/{}/clean_train_results/{}.csv'.format(
            settings.dataset, project_utils.get_optim_name(optimizer))

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    if settings.LA:
        df = pd.concat([df, pd.Series(results, name="lr={},steps={},alpha={}".format(
            settings.lr, settings.la_steps, settings.la_alpha))], axis=1)
    else:
        df = pd.concat([df, pd.Series(results, name="lr={}".format(
            settings.lr))], axis=1)

    df.to_csv(filename, index=False)


def save_valid_results(settings, optimizer, scores, attack=None):
    """
    saves validation results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        scores(list): list of test accuraries computed after every epoch
        attack(str): Optional: string containing the name attack performed during testing
        """

    if settings.adv_train:
        if attack is None:
            # create directories if necessary
            Path("Hyperparam_tuning/{}/adv_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/adv_valid_results/{}.csv'.format(settings.dataset,
                                                                             project_utils.get_optim_name(optimizer))
        else:
            Path("Hyperparam_tuning/{}/adv_{}_valid_results".format(settings.dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/adv_{}_valid_results/{}.csv'.format(settings.dataset, attack,
                                                                                project_utils.get_optim_name(optimizer))

    else:
        if attack is None:
            Path("Hyperparam_tuning/{}/clean_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/clean_valid_results/{}.csv'.format(settings.dataset,
                                                                               project_utils.get_optim_name(optimizer))
        else:
            Path("Hyperparam_tuning/{}/clean_{}_valid_results".format(settings.dataset,attack)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/clean_{}_valid_results/{}.csv'.format(settings.dataset, attack,
                                                                                  project_utils.get_optim_name(
                                                                                      optimizer))

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    if settings.LA:
        df = pd.concat([df, pd.Series(scores, name="lr={},steps={},alpha={}".format(
            settings.lr, settings.la_steps, settings.la_alpha))], axis=1)
    else:
        df = pd.concat([df, pd.Series(scores, name="lr={}".format(
            settings.lr))], axis=1)

    df.to_csv(filename, index=False)


def save_test_results(settings, optimizer, scores, attack=None):
    """
    saves results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training
        scores(pd.Dataframe): Dataframe containing the accuracy scores
        for each optimizer if a pgd attack has been executed, else None
        attack(str): Optional: string containing the name attack performed during testing
    """

    if settings.adv_train:
        if attack is None:
            # create directories if necessary
            Path("Hyperparam_tuning/{}/adv_test_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/adv_test_results/{}.csv'.format(settings.dataset,
                                                                             project_utils.get_optim_name(optimizer))
        else:
            Path("Hyperparam_tuning/{}/adv_{}_test_results".format(settings.dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/adv_{}_test_results/{}.csv'.format(settings.dataset, attack,
                                                                             project_utils.get_optim_name(optimizer))

    else:
        if attack is None:
            Path("Hyperparam_tuning/{}/clean_test_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/clean_test_results/{}.csv'.format(settings.dataset,
                                                                               project_utils.get_optim_name(optimizer))
        else:
            Path("Hyperparam_tuning/{}/clean_{}_test_results".format(settings.dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = 'Hyperparam_tuning/{}/clean_{}_test_results/{}.csv'.format(settings.dataset, attack,
                                                                                project_utils.get_optim_name(optimizer))
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, scores], axis=0)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
