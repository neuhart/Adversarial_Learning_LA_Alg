import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib
from pathlib import Path
markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
attack = 'pgd'

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
    inner_optims = ['SGD', 'Adam', 'OGD', 'ExtraSGD', 'ExtraAdam']

    for optim in inner_optims:
        df = pd.read_csv(valid_path + "/" + optim + ".csv")
        df_LA = pd.read_csv(valid_path + "/Lookahead-" + optim + ".csv")
        avg = pd.concat([avg, df.mean(axis=1).rename(optim)], axis=1)
        avg = pd.concat([avg, df_LA.mean(axis=1).rename("LA-{}".format(optim))], axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        tsplot(ax, df, marker='x',markevery=5)
        tsplot(ax, df_LA, marker='o', markevery=5)

        ax.set_xlabel('Epochs', fontsize=18)
        ax.set_ylabel('{} Validation Accuracy'.format(attack.upper()) if attack is not None else "Validation Accuracy",
                      fontsize=18)
        #ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        font = {'size': 15}
        matplotlib.rc('font', **font)
        matplotlib.pyplot.tight_layout()

        plt.legend([optim, "LA-{}".format(optim)], prop={'size': 15})
        plt.savefig("Analysis/{}/{}.png".format(dataset, optim))
        plt.show()

    avg.to_csv("Analysis/{}/avg_{}_valid_acc.csv".format(dataset, attack), index=False)


def tsplot(ax, data, marker, markevery, **kw):
    """Plots Mean Validation Accuracy with Standard deviation
    Arguments:
        ax(pyplot.axis): object where the plot is to be stored
        data(pd.Dataframe): Dataframe containing the data
        marker(str): pyplot marker
        markevery(int): determines the gap between every mark
    Returns:
        mean, std: mean and standard deviation of data
    """
    x = np.arange(1,25+1)
    mean = np.mean(data.iloc[:25,:], axis=1).rename('Mean')
    std = np.std(data.iloc[:25,:], axis=1).rename('Std')
    print('mean: {} \n standard deviation: {}'.format(mean, std))
    cis = (mean - std, mean + std)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x,mean, marker=marker, markevery=markevery)
    ax.margins(x=0)
    return mean, std

if __name__ == "__main__":
    plot_avg_valid_results()