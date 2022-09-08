import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')


def parameter_formatting(index):
    """
    Arguments:
        index(pandas.core.indexes.base.Index): iterable object containing the hyperparameter settings
    Returns:
        list of formatted hyperparameter settings
    """
    return [i.replace('alpha', '\u03B1').replace('steps','k').replace('lr','\u03B3') for i in index]


def two_scales(ax1,ax2,x_data, data1, data2, series):
    """Creates a plot with two y-axis
    Arguments:
        ax1: first axis
        ax2: second axis
        x_data: range of inputs (x-values)
        data1: y-values for first axis
        data2: y-values for second axis
        series(pd.Series): names of data series (used for legend)
    """
    for i, setting in enumerate(series.index):
        ax1.plot(x_data, data1[setting], linestyle='solid', marker=markers[i], markevery=5)
        ax2.plot(x_data, data2[setting], linestyle='dashed', marker=markers[i], markevery=5)
    start, end = ax2.get_ylim()
    ax1.set_ylim(0,1.1)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_yticks(np.arange(0, end + 1000, 2e4))
    ax1.legend(parameter_formatting(series.index))


def get_indices(file):
    col = 1 if 'Lookahead' in file else 0
    if 'ExtraAdam' in file:
        row = 4
    elif 'ExtraSGD' in file:
        row = 3
    elif 'OGD' in file:
        row = 2
    elif 'Adam' in file:
        row = 1
    elif 'SGD' in file:
        row = 0
    else:
        assert 'Wrong optimizer!'
    return row,col


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
    x = np.arange(1,data.shape[0]+1)
    mean = np.mean(data, axis=1).rename('Mean')
    std = np.std(data, axis=1).rename('Std')
    print('mean: {} \n standard deviation: {}'.format(mean, std), type(mean))
    cis = (mean - std, mean + std)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x,mean, marker=marker, markevery=markevery)
    ax.margins(x=0)
    return mean, std