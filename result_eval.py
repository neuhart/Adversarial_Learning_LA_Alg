import pandas as pd
import matplotlib.pyplot as plt
from Utils import project_utils
import os
import seaborn as sns

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = project_utils.query_dataset()
adv_results = project_utils.yes_no_check('Evaluate adversarial results?')

if adv_results:
    train_path = "results/{}/adv_train_results".format(dataset)
    test_filename = "results/{}/adv_test_results.csv".format(dataset)
else:
    train_path = "results/{}/clean_train_results".format(dataset)
    test_filename = "results/{}/clean_test_results.csv".format(dataset)


def get_avg_train_result():
    """Returns average of train results"""
    avg = pd.DataFrame()

    for file in os.listdir(train_path):
        df = pd.read_csv(train_path+"/"+file)

        avg = pd.concat([avg, df.mean(axis=1).rename(file.replace('.csv', ''))], axis=1)
    return avg


def plot_train_results(df):
    """Plots train results
    Arguments:
        df(pandas.Dataframe): Dataframe containing columns of training loss
        per optimizer
    """
    for i, col in enumerate(df.columns):
        plt.plot(range(1,df.shape[0]+1), df[col], linestyle='dashed', marker=markers[i], markevery=10)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
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


avg_train_results = get_avg_train_result()
plot_train_results(avg_train_results)
avg_test_results = get_avg_test_result()
plot_test_results(avg_test_results)
