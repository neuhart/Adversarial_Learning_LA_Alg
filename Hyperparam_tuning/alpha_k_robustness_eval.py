import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
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

optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']


def la_steps_aggregation():
    """
    Creates a validation accuracy plot for each Lookahead optimizer
    accuracies for each la_steps parameter averaged over all
    lr and la_alpha values used in the gridsearch
    """
    la_steps_list = [5,10,15]

    for optim in optims:
        df_nolook = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, la_steps in enumerate(la_steps_list):
            df2 = pd.DataFrame()
            for col in df.columns:
                if 'steps={}'.format(la_steps) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            plt.plot(range(1, df.shape[0] + 1), df2.mean(axis=1), marker=markers[i], markevery=5)

        plt.plot(range(1, df_nolook.shape[0]+1), df_nolook[df_nolook.iloc[-1].idxmax()],
                 linestyle='dashed', marker=markers[-1], markevery=5)
        plt.plot(range(1, df_nolook.shape[0]+1), df_nolook[df_nolook.iloc[-1].idxmin()],
                 linestyle='dashed', marker=markers[-2], markevery=5)

        plt.legend(['k={}'.format(la_steps) for la_steps in la_steps_list]+['{}: {}'.format(optim, df_nolook.iloc[-1].idxmax().replace('lr','\u03B3')), '{}: {}'.format(optim, df_nolook.iloc[-1].idxmin().replace('lr','\u03B3'))], loc='lower right')
        plt.title('Lookahead-{}'.format(optim))
        plt.show()


def la_alpha_aggregation():
    """
    Creates a validation accuracy plot for each Lookahead optimizer
    accuracies for each la_alpha parameter averaged over all
    lr and la_steps values used in the gridsearch
    """
    la_alphas = [0.5, 0.75, 0.9]

    for optim in optims:
        df_nolook = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, la_alpha in enumerate(la_alphas):
            df2 = pd.DataFrame()
            for col in df.columns:
                if 'alpha={}'.format(la_alpha) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            plt.plot(range(1, df.shape[0] + 1), df2.mean(axis=1), marker=markers[i], markevery=5)

        plt.plot(range(1, df_nolook.shape[0] + 1), df_nolook[df_nolook.iloc[-1].idxmax()], linestyle='dashed',
                 marker=markers[-1], markevery=5)
        plt.plot(range(1, df_nolook.shape[0] + 1), df_nolook[df_nolook.iloc[-1].idxmin()], linestyle='dashed',
                 marker=markers[-2], markevery=5)

        plt.legend(['\u03B1={}'.format(la_alpha) for la_alpha in la_alphas]+['{}: {}'.format(optim, df_nolook.iloc[-1].idxmax().replace('lr','\u03B3')), '{}: {}'.format(optim, df_nolook.iloc[-1].idxmin().replace('lr','\u03B3'))], loc='lower right')
        plt.title('Lookahead-{}'.format(optim))
        plt.show()