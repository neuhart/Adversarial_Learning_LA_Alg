import torch
import torchvision.models
from Utils import data_utils, project_utils
from Models import models, data_transformations
import training
import evaluation
import pandas as pd
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import time
from evaluation import evaluation
from pathlib import Path
import torchvision
from easydict import EasyDict


def main():
    settings = project_utils.g_settings()  # specify general settings
    settings.device = torch.device(project_utils.query_int('Select GPU [0,3]:')) if torch.cuda.is_available() else torch.device(
        'cpu')

    settings.dataset = project_utils.query_dataset()
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=data_transformations.standard_transform())

    # query which optimizers to use for training
    optims_list = project_utils.get_optims()
    if any([0 if optim.startswith('LA-') else 1 for optim in optims_list]):
        raise 'Only implemented for Lookahead'

    test_results = pd.DataFrame()

    for optim in optims_list:
        net = torchvision.models.resnet18() if settings.dataset == 'CIFAR10' else models.MNIST_CNN()
        net.to(settings.device)  # transfers to gpu if available

        # Determine which optimizer to use
        optimizer = project_utils.set_optim(optim=optim, model=net)

        # Train model
        net.train()
        training.train(settings, data.train, net, optimizer)

        # Evaluation
        net.eval()
        results = evaluation.evaluation(settings, data.test, net)

        test_results = pd.concat([test_results, pd.Series(results, name=project_utils.get_optim_name(optimizer))],
                                 axis=1)
    project_utils.save_test_results(settings.dataset, settings.adv_train, test_results)


def train(settings, data, model, optimizer):
    """trains the network on the provided training set using the given optimizer
    Arguments:
        settings(EasyDict): easydict dictionary containing the training settings
        data(EasyDict): easydict dict containing dataloaders for training and testing
        model(torch.nn.Module): model to be trained
        optimizer(torch.optim.Optimizer): optimizer used to train the model
    """

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    slow_weights_valid_results = []
    fast_weights_valid_results = []

    for epoch in range(1, settings.nb_epochs + 1):
        start_t = time.time()
        for x, y in data.train:
            x, y = x.to(settings.device), y.to(settings.device)
            if settings.adv_train:
                x = projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf)
                x = x.detach()

            if project_utils.get_optim_name(optimizer) in ['Lookahead-ExtraAdam', 'Lookahead-ExtraSGD']:
                # For LA-Extra Algs we need to perform an extrapolation step with the inner optimizer
                optimizer.optimizer.extrapolation()
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            if epoch in [2,6,12]:
                # Validation on slow weights
                optimizer._backup_and_load_cache()
                model.eval()
                slow_weights_valid_results.append(evaluation(settings, data.test, model).clean)
                model.train()
                optimizer._clear_and_load_backup()

                # Validation on fast weights
                model.eval()
                fast_weights_valid_results.append(evaluation(settings, data.test, model).clean)
                model.train()

                save_valid_results(settings, optimizer, scores=slow_weights_valid_results, weights='slow')
                save_valid_results(settings, optimizer, scores=fast_weights_valid_results, weights='fast')

        end_t = time.time()
        print(
            "epoch: {}/{} computed in {:.3f} seconds".format(
                epoch, settings.nb_epochs, end_t-start_t
            )
        )


def save_valid_results(settings, optimizer, scores, weights):
    """
    saves validation results to csv-file
    Arguments:
        settings(EasyDict): easydict dictionary containing settings and hyperparameters
        optimizer(torch.optim.Optimizer): optimizer used for training the model
        scores(list): list of test accuraries computed after every epoch
        weights(str): states on which weights (i.e. fast or slow) validation has been performed
        """
    # create directories if necessary
    Path("slow_weight_evaluation/{}/adv_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)
    Path("slow_weight_evaluation/{}/clean_valid_results".format(settings.dataset)).mkdir(parents=True, exist_ok=True)

    if settings.adv_train:
        filename = 'slow_weight_evaluation/{}/adv_valid_results/{}-{}-weights.csv'.format(
            settings.dataset, project_utils.get_optim_name(optimizer), weights)
    else:
        filename = 'slow_weight_evaluation/{}/clean_valid_results/{}_{}-weights.csv'.format(
            settings.dataset, project_utils.get_optim_name(optimizer), weights)

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(scores, name='lr={};steps={};alpha={}'.format(
        optimizer.optimizer.param_groups[0]['lr'], optimizer.la_total_la_steps, optimizer.la_alpha))], axis=1)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
