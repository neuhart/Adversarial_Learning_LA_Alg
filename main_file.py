import torch
from Utils import data_utils, project_utils
from Models import models
import training
from easydict import EasyDict


def main():
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=settings.transform)

    for i in range(settings.nb_runs):

        for optim_name in settings.optim_list:
            net = settings.model
            net.to(settings.device)  # transfers to gpu if available

            settings.hyperparams = project_utils.get_opt_hyperprams(settings, optim_name)
            # Instantiate optimizer
            optimizer = project_utils.set_optim(settings, optim=optim_name, model=net)

            # Train model (including validation)
            net.train()
            training.train(settings, data, net, optimizer)


if __name__ == "__main__":
    settings = EasyDict(
        dataset=project_utils.query_dataset(),
        optim_list=project_utils.get_optims(),
        adv_train=project_utils.yes_no_check('Adversarial Training?'),
        nb_runs=project_utils.query_int('Number of runs'),
        nb_epochs=project_utils.query_int('Number of epochs'),
        device=torch.device(project_utils.query_int('Select GPU [0,3]:')) if \
            torch.cuda.is_available() else torch.device('cpu'),
        save_to_folder='results'
    )
    settings = models.set_model_and_transform(settings)  # instantiate model & set transform

    main()
