import pandas as pd
import matplotlib.pyplot as plt
import os
from hyper_param_tuning_eval import parameter_formatting
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
from pathlib import Path
from Utils.visualization_utils import *


markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
adv_train = True
clean_test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
fgsm_test_path = "{}/adv_fgsm_test_results".format(dataset) if adv_train else "{}/clean_fgsm_test_results".format(dataset)
pgd_test_path = "{}/adv_pgd_test_results".format(dataset) if adv_train else "{}/clean_pgd_test_results".format(dataset)

attack = None
clean_valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
fgsm_valid_path = "{}/adv_fgsm_valid_results".format(dataset) if adv_train else "{}/clean_fgsm_valid_results".format(dataset)
pgd_valid_path = "{}/adv_pgd_valid_results".format(dataset) if adv_train else "{}/clean_pgd_valid_results".format(dataset)

train_path = "{}/adv_train_results".format(dataset) if adv_train else "{}/clean_train_results".format(dataset)
optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']


def valid_acc():
    """Plot of model accuracies validated on clean,fgsm and pgd images of top3 hyperparameter settings"""

    for file in os.listdir(clean_valid_path):
        if not file.startswith('Lookahead'):
            fig, ax = plt.subplots(3, 2, figsize=(12,10))

            df_clean = pd.read_csv(clean_valid_path + "/" + file)
            df_fgsm = pd.read_csv(fgsm_valid_path + "/" + file)
            df_pgd = pd.read_csv(pgd_valid_path + "/" + file)

            df_LA_clean = pd.read_csv(clean_valid_path + "/Lookahead-" + file)
            df_LA_fgsm = pd.read_csv(fgsm_valid_path + "/Lookahead-" + file)
            df_LA_pgd = pd.read_csv(pgd_valid_path + "/Lookahead-" + file)


            # top 3 settings regarding acc on clean images
            top3_clean_valid_series = df_clean.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_clean_valid_series = df_LA_clean.iloc[-1].sort_values(ascending=False)[:3]
            # top 3 settings regarding acc on fgsm perturbed images
            top3_fgsm_valid_series = df_fgsm.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_fgsm_valid_series = df_LA_fgsm.iloc[-1].sort_values(ascending=False)[:3]
            # top 3 settings regarding acc on pgd perturbed images
            top3_pgd_valid_series = df_pgd.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_pgd_valid_series = df_LA_pgd.iloc[-1].sort_values(ascending=False)[:3]

            for i, col in enumerate(top3_clean_valid_series.index):
                ax[0,0].plot(range(1, df_clean.shape[0] + 1), df_clean[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_clean_valid_series.index):
                ax[0, 1].plot(range(1, df_LA_clean.shape[0] + 1), df_LA_clean[col], marker=markers[i], markevery=5)

            for i, col in enumerate(top3_fgsm_valid_series.index):
                ax[1,0].plot(range(1, df_fgsm.shape[0] + 1), df_fgsm[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_fgsm_valid_series.index):
                ax[1, 1].plot(range(1, df_LA_fgsm.shape[0] + 1), df_LA_fgsm[col], marker=markers[i], markevery=5)

            for i, col in enumerate(top3_pgd_valid_series.index):
                ax[2,0].plot(range(1, df_pgd.shape[0] + 1), df_pgd[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_pgd_valid_series.index):
                ax[2,1].plot(range(1, df_LA_pgd.shape[0] + 1), df_LA_pgd[col], marker=markers[i], markevery=5)

            for i in range(3):
                for j in range(2):
                    ax[i,j].set_ylim([0,1.1])

            legend_loc = 'upper left' if dataset == 'CIFAR10' else 'lower right'
            ax[0,0].legend(parameter_formatting(top3_clean_valid_series.index), loc=legend_loc)
            ax[0,1].legend(parameter_formatting(top3_LA_clean_valid_series.index), loc=legend_loc)
            ax[1,0].legend(parameter_formatting(top3_fgsm_valid_series.index), loc=legend_loc)
            ax[1,1].legend(parameter_formatting(top3_LA_fgsm_valid_series.index), loc=legend_loc)
            ax[2,0].legend(parameter_formatting(top3_pgd_valid_series.index), loc=legend_loc)
            ax[2,1].legend(parameter_formatting(top3_LA_pgd_valid_series.index), loc=legend_loc)

            ax[2,0].set_xlabel('Epochs')
            ax[2, 1].set_xlabel('Epochs')
            ax[0, 0].set_ylabel('Clean Accuracy')
            ax[1, 0].set_ylabel('FGSM Accuracy')
            ax[2, 0].set_ylabel('PGD Accuracy')
            ax[0,0].set_title('{}'.format(file.replace('.csv', '')))
            ax[0,1].set_title('Lookahead-{}'.format(file.replace('.csv', '')))
            plt.show()



def train_loss_vs_valid_acc():
    """
    Creates 2x2 Plot of validation acc and training loss for optimizers and
    their Lookahead version to allow for comparisons. Top 3 and Bottom 3 settings are plotted
    """

    for optim in optims:
        fig, ax = plt.subplots(2, sharey='all', figsize=(10,10))

        valid_df = pd.read_csv(pgd_valid_path + "/" + '{}.csv'.format(optim))
        train_df = pd.read_csv(train_path + "/" + '{}.csv'.format(optim))
        top3_series = valid_df.iloc[-1].sort_values(ascending=False)[:3]  # top 5 settings
        ax2 = ax[0].twinx()
        two_scales(ax[0], ax2, range(1, valid_df.shape[0] + 1),
                                         valid_df, train_df, top3_series)
        ax2.set_ylabel('Training Loss')

        valid_df_Lookahead = pd.read_csv(pgd_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        train_df_Lookahead = pd.read_csv(train_path + "/" + 'Lookahead-{}.csv'.format(optim))
        top3_LA_series = valid_df_Lookahead.iloc[-1].sort_values(ascending=False)[:3]  # top 5 settings

        ax2 = ax[1].twinx()
        two_scales(ax[1], ax2, range(1, valid_df_Lookahead.shape[0] + 1),
                   valid_df_Lookahead, train_df_Lookahead, top3_LA_series)


        ax2.set_ylabel('Training Loss')
        ax[0].set_title('{}'.format(optim))
        ax[1].set_title('Lookahead-{}'.format(optim))
        ax[0].set_ylabel('PGD Validation Accuracy')
        ax[1].set_ylabel('PGD Validation Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs')
        Path("Analysis/{}/adv_top3/".format(dataset)).mkdir(parents=True, exist_ok=True)
        plt.savefig("Analysis/{}/adv_top3/{}.png".format(dataset, optim))
        plt.show()

train_loss_vs_valid_acc()