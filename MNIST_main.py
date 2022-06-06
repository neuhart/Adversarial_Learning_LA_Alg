import pandas as pd
import torch
import project_utils
import data_utils
from Models import models, data_transformations
import training
import evaluation


def main():
    settings = project_utils.g_settings()  # specify general settings
    settings.device = torch.device(project_utils.query_int('Select GPU [0,3]:')) if \
        torch.cuda.is_available() else torch.device('cpu')

    settings.dataset = 'MNIST'
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=data_transformations.standard_transform())

    # query which optimizers to use for training
    optims_list = project_utils.get_optims()

    test_results = pd.DataFrame()

    for optim in optims_list:

        net = models.MNIST_CNN()
        net.to(settings.device)  # transfers to gpu if available

        # Determine which optimizer to use
        optimizer = project_utils.set_optim(optim=optim, model=net)

        # Train model
        net.train()
        training.train(settings, data.train, net, optimizer)

        # Evaluation
        net.eval()
        results = evaluation.evaluation(settings, data.test, net)

        test_results = pd.concat([test_results, pd.Series(results, name=project_utils.get_optim_name(optimizer))], axis=1)
    project_utils.save_test_results(settings.dataset, settings.adv_train, test_results)


if __name__ == "__main__":
    main()
