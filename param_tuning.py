from Utils import data_utils, project_utils
from Optimizer.Lookahead import Lookahead
from Optimizer.OGDA import OGDA
from Optimizer.extragradient import ExtraSGD, ExtraAdam
from easydict import EasyDict
import torch
import training
from Models import models


def main():
    """Performs hyperparameter tuning on a selected dataset for all implemented optimizers.
    Saves train loss and validation accduracy to the Hyperparam_tuning folder."""
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=settings.transform)

    for i in range(settings.nb_runs):
        for optim_name in settings.optim_list:
            for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
                if optim_name.startswith('Lookahead-'):  # check if Lookahead is used
                    inner_optim_name = optim_name[10:]  # delete 'Lookahead-' prefix

                    for la_steps in [5, 10]:
                        for la_alpha in [0.5, 0.75, 0.9]:
                            settings.hyperparams = (lr, la_steps, la_alpha)

                            # select correct neural network
                            net = settings.model
                            net.to(settings.device)

                            # Instantiate optimizer
                            optimizer = Lookahead(set_inner_optim(
                                inner_optim_name, lr=lr, model=net), la_steps=la_steps, la_alpha=la_alpha)

                            # Train model
                            net.train()
                            training.train(settings, data, net, optimizer)

                else:
                    settings.hyperparams = (lr, 0, 0)
                    net = settings.model
                    net.to(settings.device)  # transfers to gpu if available

                    # Determine which optimizer to use
                    optimizer = set_inner_optim(optim_name, lr=lr, model=net)

                    # Train model
                    net.train()
                    training.train(settings, data, net, optimizer)


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


if __name__ == "__main__":
    settings = EasyDict(dataset=project_utils.query_dataset(),
                        optim_list=project_utils.get_optims(),
                        adv_train=project_utils.yes_no_check('Adversarial Training?'),
                        nb_runs=project_utils.query_int('Number of runs?'),
                        nb_epochs=project_utils.query_int('Number of epochs?'),
                        device=torch.device(project_utils.query_int('Select GPU [0,3]:')) if \
                        torch.cuda.is_available() else torch.device('cpu'),
                        save_to_folder='Hyperparam_tuning'
                        )
    settings = models.set_model_and_transform(settings)
    main()
