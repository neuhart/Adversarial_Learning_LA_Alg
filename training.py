import torch
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import time
import project_utils


def train(settings, train_loader, model, optimizer):
    """trains the network on the provided training set using the given optimizer
    Arguments:
        settings(EasyDict): easydict dictionary containing the training settings
        train_loader(torch Dataloader): Dataloader used for Mini-batching
        model(torch.nn.Module): model to be trained
        optimizer(torch.optim.Optimizer): optimizer used to train the model
    """

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    scheduler = project_utils.set_lr_scheduler(optimizer)

    results = []
    for epoch in range(1, settings.nb_epochs + 1):
        start_t = time.time()
        train_loss = 0.0
        for x, y in train_loader:
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

        scheduler.step()

        end_t = time.time()
        print(
            "epoch: {}/{}, train loss: {:.3f} computed in {:.3f} seconds".format(
                epoch, settings.nb_epochs, train_loss/len(train_loader), end_t-start_t
            )
        )
        results.append(train_loss)
    project_utils.save_train_results(optimizer, dataset=settings.dataset, adv_train=settings.adv_train, results=results)
