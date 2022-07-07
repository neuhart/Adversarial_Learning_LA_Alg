import torch
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import time
from Utils import project_utils
from easydict import EasyDict
from evaluation import evaluation


def train(settings, data, model, optimizer):
    """trains the network on the provided training set using the given optimizer
    Arguments:
        settings(EasyDict): easydict dictionary containing the training settings
        data(EasyDict): easydict dict containing dataloaders for training and testing
        model(torch.nn.Module): model to be trained
        optimizer(torch.optim.Optimizer): optimizer used to train the model
    """

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    train_results = []
    valid_clean_results = []
    if settings.fgsm_att: valid_fgsm_results = []
    if settings.pgd_att: valid_pgd_results = []

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
        validation_results = evaluation(settings, data.test, model)
        valid_clean_results.append(validation_results.clean)
        if settings.fgsm_att: valid_fgsm_results.append(validation_results.fgsm_att)
        if settings.pgd_att: valid_pgd_results.append(validation_results.pgd_att)
        model.train()

    project_utils.save_train_results(settings, optimizer, results=train_results)
    project_utils.save_valid_results(settings, optimizer, scores=valid_clean_results)
    if settings.fgsm_att:
        project_utils.save_valid_results(settings, optimizer, scores=valid_fgsm_results, attack='fgsm')
    if settings.pgd_att:
        project_utils.save_valid_results(settings, optimizer, scores=valid_fgsm_results, attack='pgd')


