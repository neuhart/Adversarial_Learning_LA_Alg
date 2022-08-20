import torch
from Utils import data_utils, project_utils
from Models import models, data_transformations
import training
from easydict import EasyDict
import torchvision


def main():
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=settings.transform)

    for i in range(settings.nb_runs):

        for optim_name in settings.optim_list:
            net = settings.model
            net.to(settings.device)  # transfers to gpu if available

            # Determine which optimizer to use
            optimizer = project_utils.set_optim(settings, optim=optim_name, model=net)

            # Train model (including validation)
            net.train()
            training.train(settings, data, net, optimizer)


if __name__ == "__main__":
    settings = EasyDict(
        nb_runs=project_utils.query_int('Number of runs'),
        nb_epochs=project_utils.query_int('Number of epochs'),
        adv_train=project_utils.yes_no_check('Adversarial Training?'),
        device=torch.device(project_utils.query_int('Select GPU [0,3]:')) if \
            torch.cuda.is_available() else torch.device('cpu'),
        dataset=project_utils.query_dataset(),
        optim_list=project_utils.get_optims()
    )
    if settings.dataset == 'MNIST':
        settings.model = models.MNIST_CNN()
        settings.transform=data_transformations.standard_transform()
    elif settings.dataset == 'FashionMNIST':
        settings.transform = data_transformations.standard_transform()
        settings.model = models.MNIST_CNN()
    elif settings.dataset == 'CIFAR10':
        settings.transform = data_transformations.resnet_transform()
        settings.model = torchvision.models.resnet18()

    main()
