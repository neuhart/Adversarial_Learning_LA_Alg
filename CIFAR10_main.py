import torch
import torchvision.models
from Utils import data_utils, project_utils
from Models import data_transformations
import training
import evaluation
import pandas as pd


def main():
    data = data_utils.ld_dataset(dataset_name=settings.dataset, transform=data_transformations.resnet_transform())

    for i in range(project_utils.query_int('Number of runs')):

        clean_scores = pd.DataFrame()
        fgsm_scores = pd.DataFrame()
        pgd_scores = pd.DataFrame()

        for optim in optims_list:
            net = torchvision.models.resnet18()
            net.to(settings.device)  # transfers to gpu if available

            # Determine which optimizer to use
            optimizer = project_utils.set_optim(settings, optim=optim, model=net)

            # Train model
            net.train()
            training.train(settings, data, net, optimizer)

            # Evaluation
            net.eval()
            results = evaluation.evaluation(settings, data.test, net)

            clean_scores = pd.concat([clean_scores, pd.Series(results, name=project_utils.get_optim_name(optimizer))],
                                     axis=1)
            fgsm_scores = pd.concat([fgsm_scores, pd.Series(results.fgsm_att,
                                                                name=project_utils.get_optim_name(optimizer))], axis=1)

            pgd_scores = pd.concat([pgd_scores, pd.Series(results.pgd_att,
                                                              name=project_utils.get_optim_name(optimizer))], axis=1)

        project_utils.save_test_results(settings, clean_scores)
        project_utils.save_test_results(settings, fgsm_scores, attack='fgsm')
        project_utils.save_test_results(settings, pgd_scores, attack='pgd')


if __name__ == "__main__":
    settings = project_utils.g_settings()  # specify general settings
    settings.device = torch.device(
        project_utils.query_int('Select GPU [0,3]:')) if torch.cuda.is_available() else torch.device(
        'cpu')

    settings.dataset = 'CIFAR10'
    # query which optimizers to use for training
    optims_list = project_utils.get_optims()

