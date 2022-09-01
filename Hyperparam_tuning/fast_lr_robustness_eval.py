import pandas as pd
import matplotlib.pyplot as plt
import os
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


def lr_aggregation_summaryplot():
    """
    Creates a 5x2 figure of validation accuracies.
    In each row, in the left plot, a validation accuracy plot is created for an optimizer,
    and in the right plot, a valid. acc. plot,
    averaged over all values of la_steps and la_alpha used in the gridsearch,
    is created for the Lookahead version of the optimizer
    """
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    fig, ax = plt.subplots(5, 2, sharey='all')
    fig.set_figheight(20)
    fig.set_figwidth(13)
    fig.suptitle('LR Aggregation')
    for file in os.listdir(clean_valid_path):
        r, c = get_indices(file)
        df = pd.read_csv(clean_valid_path + "/" + file)
        for i, lr in enumerate(learning_rates):

            df2 = pd.DataFrame()
            for col in df.columns:
                if 'lr={}'.format(lr) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            ax[r,c].plot(range(1, df.shape[0] + 1), df2.mean(axis=1),  linestyle='dashed', marker=markers[i], markevery=5)

        ax[r,c].set_title(file.replace('.csv',''))
        ax[r, c].legend(['\u03B3={}'.format(lr) for lr in learning_rates])
    plt.show()


def lr_aggregation_pairplot():
    """Creates a 1x2 figure consisting of two plots.
    On the left: a plot of the validation accuracy for a given optimizer
    On the right: a plot of the valid. acc. of Lookahead version of the optimizer
    averaged over all la_steps and la_alpha used in the gridsearch
    """
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]

    for optim in optims:
        fig, ax = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(5)
        fig.set_figwidth(10)
        df = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df_Lookahead = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, lr in enumerate(learning_rates):
            df2 = pd.DataFrame()
            df2_Lookahead = pd.DataFrame()
            for col in df.columns:
                if 'lr={}'.format(lr) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)
            for col in df_Lookahead.columns:
                if 'lr={}'.format(lr) in col:
                    df2_Lookahead = pd.concat([df2_Lookahead, df_Lookahead[col]], axis=1)

            ax[0].plot(range(1, df.shape[0] + 1), df2.mean(axis=1),  linestyle='dashed', marker=markers[i], markevery=5)
            ax[1].plot(range(1, df.shape[0] + 1), df2_Lookahead.mean(axis=1), linestyle='dashed', marker=markers[i],
                          markevery=5)
        ax[0].set_title('{}'.format(optim))
        ax[1].set_title('Lookahead-{}'.format(optim))
        ax[0].set_ylabel('Accuracy')
        ax[1].set_ylabel('Avg. Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs')
        plt.legend(['\u03B3={}'.format(lr) for lr in learning_rates], loc='lower right')
        plt.show()


def lr_avg_acc():
    """
    Plots PGD validation accurcary for all optimizers for each (fast) learning rate on a given data set averaged over
    hyperparameters k and alpha) used in the gridsearch (see param_tuning.py) and saves results (mean+std figures to csv and plots to png)
    """
    if attack is None:
        source_path = clean_valid_path
    elif attack == 'fgsm':
        source_path = fgsm_valid_path
    elif attack == 'pgd':
        source_path = pgd_valid_path
    else:
        raise 'Attack "{}" not implemented'.format(attack)

    for optim_name in optims:
        fig, ax = plt.subplots(1)
        df = pd.read_csv(source_path+'/' + optim_name + '.csv')
        df_LA = pd.read_csv(source_path+'/Lookahead-' + optim_name + '.csv')

        mean_series, std_series = tsplot(ax, df, markers[0], 5)

        if attack is None:
            Path("Analysis/{}/LR_robustness/adv_valid_results_mean_std".format(dataset)).mkdir(parents=True, exist_ok=True)
            filename = "Analysis/{}/LR_robustness/adv_valid_results_mean_std/{}.csv".format(dataset, optim_name)
        else:
            Path("Analysis/{}/LR_robustness/adv_{}_valid_results_mean_std".format(dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = "Analysis/{}/LR_robustness/adv_{}_valid_results_mean_std/{}.csv".format(
                dataset, attack, optim_name)
        pd.concat([mean_series,std_series], axis=1).to_csv(filename, index=False)

        mean_series, std_series = tsplot(ax, df_LA, markers[1], 5)
        if attack is None:
            filename_LA = "Analysis/{}/LR_robustness/adv_valid_results_mean_std/{}.csv".format(dataset, 'LA-' + optim_name)
        else:
            filename_LA = "Analysis/{}/LR_robustness/adv_{}_valid_results_mean_std/{}.csv".format(
                dataset, attack, 'LA-' + optim_name)
        pd.concat([mean_series,std_series], axis=1).to_csv(filename_LA, index=False)

        ax.legend([optim_name, 'LA-' + optim_name])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        plt.savefig(filename[:-3]+'png')
        plt.show()


if __name__ == "__main__":
    lr_aggregation_pairplot()
    # lr_aggregation_summaryplot()
