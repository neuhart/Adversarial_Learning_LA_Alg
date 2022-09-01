import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
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

top_settings = pd.DataFrame()
results= pd.DataFrame()
barplots = False
optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']


def top5_plots():
    for file in os.listdir(clean_test_path):
        """Plot of test accuracies of top 5 and bottom 5 hyperparameter settings"""
        df = pd.read_csv(clean_test_path + "/" + file)

        top5_series = df.iloc[0].sort_values(ascending=False)[:5]  # top 5 settings
        bottom5_series = df.iloc[0].sort_values(ascending=True)[:5]  # bottom 5 settings

        if barplots:
            fig, ax = plt.subplots(1, 2, sharey='all')
            sns.barplot(ax=ax[0],x=top5_series.index, y=top5_series)
            sns.barplot(ax=ax[1],x=bottom5_series.index, y=bottom5_series)
            ax[0].set_ylabel('Accuracy')
            ax[1].set_ylabel('')
            # plt.xticks(rotation=90) only rotated xticks of right-hand plot
            # Workaround:
            ax[0].set_xticklabels(top5_series.index,rotation=90)
            ax[1].set_xticklabels(bottom5_series.index, rotation=90)

            ax[0].set_title('Top 5')
            ax[1].set_title('Bottom 5')
            fig.suptitle('{}'.format(file.replace('.csv', '')))
            plt.tight_layout()
            plt.show()

        """Plot of validation accuracies of top5 hyperparameter settings"""
        df = pd.read_csv(clean_valid_path + "/" + file)
        top5_valid_series = df.iloc[-1].sort_values(ascending=False)[:5]  # top 5 settings

        for i, col in enumerate(top5_valid_series.index):
            plt.plot(range(1, df.shape[0] + 1), df[col], linestyle='dashed', marker=markers[i], markevery=5)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(parameter_formatting(top5_series.index))
        plt.title('{}'.format(file.replace('.csv', '')))
        plt.show()

