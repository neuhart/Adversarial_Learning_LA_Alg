import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'


def parameter_formatting(hyperparameter_string):
    """
    Arguments:
        hyperparameter_string(str): string containing the hyperparameter settings
    Returns:
        string of formatted hyperparameter settings
    """
    return hyperparameter_string.replace('alpha', '\u03B1').replace('steps','k').replace('lr','\u03B3')


for optim_name in ['Lookahead-SGD','Lookahead-Adam','Lookahead-OGD', 'Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']:

    fast_weights_valid_path = "{}/adv_valid_results/{}-fast-weights.csv".format(dataset, optim_name)
    fast_df = pd.read_csv(fast_weights_valid_path)

    fig, ax = plt.subplots(1, 2, sharey='all', figsize=(15, 5))

    for col in range(fast_df.shape[1]):
        fast_series= fast_df.iloc[:,col]
        k = int(fast_series.name.split(';')[1].split('=')[1])  # get Lookahead steps parameter
        slow_series = fast_series[fast_series.index % k == 0]

        # slow weights
        ls = [slow_weight_update*(k-1) for slow_weight_update in range(0, math.floor(fast_df.shape[0]/k))]  # [0] +
        ax[col].plot(ls, slow_series,
                     'r', marker='x', linestyle='dashed')

        # fast weights
        ax[col].plot(range(0,k), fast_series.iloc[0:k], 'b')
        for j in range(1,math.floor(fast_df.shape[0]/k)):
           ax[col].plot(range(j*(k-1), j*(k-1)+k), fast_series.iloc[j*k:j*k+k], 'b')


        ax[0].legend(['slow-weights', 'fast-weights'])
        fig.suptitle(optim_name.replace('Lookahead','LA'))
        ax[col].set_title(parameter_formatting(slow_series.name))
        ax[col].set_ylabel('Accuracy')
        ax[col].set_xlabel('fast-weights update')

    Path("Analysis/{}/adv_pgd_valid_slow".format(dataset)).mkdir(parents=True, exist_ok=True)
    plt.savefig("Analysis/{}/adv_pgd_valid_slow/{}.png".format(dataset, optim_name.replace('Lookahead', 'LA')))
    plt.show()


