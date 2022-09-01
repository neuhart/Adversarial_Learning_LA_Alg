import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.visualization_utils import *


markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
adv_train = True
clean_test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
fgsm_test_path = "{}/adv_fgsm_test_results".format(dataset) if adv_train else "{}/clean_fgsm_test_results".format(dataset)
pgd_test_path = "{}/adv_pgd_test_results".format(dataset) if adv_train else "{}/clean_pgd_test_results".format(dataset)

attack = 'pgd'
clean_valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
fgsm_valid_path = "{}/adv_fgsm_valid_results".format(dataset) if adv_train else "{}/clean_fgsm_valid_results".format(dataset)
pgd_valid_path = "{}/adv_pgd_valid_results".format(dataset) if adv_train else "{}/clean_pgd_valid_results".format(dataset)

optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']


def total_avg_acc():
    """
    Plots PGD validation accurcary for all optimizers on a given data set averaged over ALL hyperparameter settings
    used in the gridsearch (see param_tuning.py) and saves results (mean+std figures to csv and plots to png)
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
            Path("Analysis/{}/overall_robustness/adv_valid_results_mean_std".format(dataset)).mkdir(parents=True, exist_ok=True)
            filename = "Analysis/{}/overall_robustness/adv_valid_results_mean_std/{}.csv".format(dataset, optim_name)
        else:
            Path("Analysis/{}/overall_robustness/adv_{}_valid_results_mean_std".format(dataset, attack)).mkdir(parents=True, exist_ok=True)
            filename = "Analysis/{}/overall_robustness/adv_{}_valid_results_mean_std/{}.csv".format(
                dataset, attack, optim_name)
        pd.concat([mean_series,std_series], axis=1).to_csv(filename, index=False)

        mean_series, std_series = tsplot(ax, df_LA, markers[1], 5)
        if attack is None:
            filename_LA = "Analysis/{}/overall_robustness/dv_valid_results_mean_std/{}.csv".format(dataset, 'LA-' + optim_name)
        else:
            filename_LA = "Analysis/{}/overall_robustness/adv_{}_valid_results_mean_std/{}.csv".format(
                dataset, attack, 'LA-' + optim_name)
        pd.concat([mean_series,std_series], axis=1).to_csv(filename_LA, index=False)

        ax.legend([optim_name, 'LA-' + optim_name])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        plt.savefig(filename[:-3]+'png')
        plt.show()


if __name__ == "__main__":
    total_avg_acc()