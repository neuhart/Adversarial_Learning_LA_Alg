import pandas as pd
import matplotlib.pyplot as plt
from Utils import project_utils
import os
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = project_utils.query_dataset()
adv_results = project_utils.yes_no_check('Evaluate adversarial results?')

if adv_results:
    train_path = "results/{}/adv_train_results".format(dataset)
    valid_path = "results/{}/adv_valid_results".format(dataset)
    test_filename = "results/{}/adv_test_results.csv".format(dataset)
else:
    train_path = "results/{}/clean_train_results".format(dataset)
    valid_path = "results/{}/clean_valid_results".format(dataset)
    test_filename = "results/{}/clean_test_results.csv".format(dataset)


def get_avg_train_result():
    """Returns average of train results"""
    avg = pd.DataFrame()

    for file in os.listdir(train_path):
        df = pd.read_csv(train_path+"/"+file)

        avg = pd.concat([avg, df.mean(axis=1).rename(file.replace('.csv', ''))], axis=1)
    return avg


def get_avg_valid_result():
    """Returns average of validation results"""
    avg = pd.DataFrame()

    for file in os.listdir(valid_path):
        df = pd.read_csv(valid_path+"/"+file)
        avg = pd.concat([avg, df.mean(axis=1).rename(file.replace('.csv', ''))], axis=1)
    return avg


def plot_results(df, df2):
    """Plots validation accuracy
    Arguments:
        df(pandas.Dataframe): Dataframe containing columns of training loss
        per optimizer
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, col in enumerate(df.columns):
        ax.plot(range(1,df2.shape[0]+1), df2[col], linestyle='dashed', marker=markers[i], markevery=5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(df.columns)
    plt.show()


def get_avg_test_result():
    """Returns:
    avg(pandas Series): average of test accuracies per optimizer"""

    df = pd.read_csv(test_filename)
    avg = df.mean(axis=0)
    return avg


def plot_test_results(s):
    """Plots train results
    Arguments:
        s(pandas.Series): Series containing averaged test accuracies
        per optimizer
    """
    sns.barplot(s.index, s.values)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def slow_weights_plot():
    """
    for file in os.listdir(test_path):
        if file.startswith('Lookahead'):
            df = pd.read_csv(valid_path + "/" + file)

            top5_series = df.iloc[0].sort_values(ascending=False)[:5]  # top 5 settings
            # plot fast and slow weights
    """


avg_train_results = get_avg_train_result()
avg_valid_results = get_avg_valid_result()
plot_results(avg_train_results, avg_valid_results)
avg_test_results = get_avg_test_result()
plot_test_results(avg_test_results)
