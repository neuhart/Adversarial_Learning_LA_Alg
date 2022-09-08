import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as mtick
from pathlib import Path
from Utils.visualization_utils import tsplot
markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
attack = None

if attack is None:
    train_path = "{}/adv_train_results".format(dataset)
    valid_path = "{}/adv_valid_results".format(dataset)
    Path("Analysis/{}/avg_valid_acc".format(
        dataset, attack)).mkdir(parents=True, exist_ok=True)
else:
    train_path = "{}/adv_{}_train_results".format(dataset, attack)
    valid_path = "{}/adv_{}_valid_results".format(dataset, attack)
    Path("Analysis/{}/avg_{}_valid_acc".format(
        dataset, attack)).mkdir(parents=True, exist_ok=True)


def plot_avg_valid_results():
    """Plots validation accuracy"""
    avg = pd.DataFrame()
    std = pd.DataFrame()
    inner_optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']

    for optim in inner_optims:
        df = pd.read_csv(valid_path + "/" + optim + ".csv")
        df_LA = pd.read_csv(valid_path + "/Lookahead-" + optim + ".csv")
        avg = pd.concat([avg, df.mean(axis=1).rename(optim)], axis=1)
        avg = pd.concat([avg, df_LA.mean(axis=1).rename("LA-{}".format(optim))], axis=1)

        std = pd.concat([std, np.std(df, axis=1).rename(optim)], axis=1)
        std = pd.concat([std, np.std(df_LA, axis=1).rename("LA-{}".format(optim))], axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        tsplot(ax, df, marker='x',markevery=5)
        tsplot(ax, df_LA, marker='o', markevery=5)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('{} Validation Accuracy'.format(attack.upper()) if attack is not None else "Validation Accuracy")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.legend([optim, "LA-{}".format(optim)])
        if attack is not None:
            plt.savefig("Analysis/{}/avg_{}_valid_acc/{}.png".format(dataset, attack, optim))
        else:
            plt.savefig("Analysis/{}/avg_valid_acc/{}.png".format(dataset, optim))
        plt.show()

    if attack is not None:
        avg.to_csv("Analysis/{}/avg_{}_valid_acc/avg.csv".format(dataset, attack), index=False)
        std.to_csv("Analysis/{}/avg_{}_valid_acc/std.csv".format(dataset, attack), index=False)
    else:
        avg.to_csv("Analysis/{}/avg_valid_acc/avg.csv".format(dataset), index=False)
        std.to_csv("Analysis/{}/avg_valid_acc/std.csv".format(dataset), index=False)


if __name__ == "__main__":
    plot_avg_valid_results()
