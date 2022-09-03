import pandas as pd
import matplotlib.pyplot as plt

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
adv_train=True


def parameter_formatting(hyperparameter_string):
    """
    Arguments:
        hyperparameter_string(str): string containing the hyperparameter settings
    Returns:
        string of formatted hyperparameter settings
    """
    return hyperparameter_string.replace('alpha', '\u03B1').replace('steps','k').replace('lr','\u03B3')


for optim in ['Lookahead-SGD','Lookahead-Adam','Lookahead-OGD', 'Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']:
    if adv_train:
        fast_weights_valid_path = "{}/adv_valid_results/{}-fast-weights.csv".format(dataset, optim)
        slow_weights_valid_path = "{}/adv_valid_results/{}-slow-weights.csv".format(dataset, optim)
    else:
        fast_weights_valid_path = "{}/clean_valid_results/{}_fast-weights.csv".format(dataset, optim)
        slow_weights_valid_path = "{}/clean_valid_results/{}_slow-weights.csv".format(dataset, optim)

    fast_df = pd.read_csv(fast_weights_valid_path)
    slow_df= pd.read_csv(slow_weights_valid_path)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(15, 5))

    for col in range(slow_df.shape[1]):
        slow_series = slow_df.iloc[:,col]
        k = int(slow_series.name.split(';')[1].split('=')[1])  # get Lookahead steps parameter

        # slow weights
        ax[col].plot([(k-1)*i+(k-1) for i in range(10)],  slow_series[(slow_series.index + 1) % k == 0][:10], 'r',  linestyle='dashed', marker='x')

        # fast weights
        ax[col].plot(range(1,k), fast_df.iloc[0:k-1, col], 'b')
        for j in range(1,10):
            ax[col].plot(range(j*(k-1), (k-1)*j+k), fast_df.iloc[k*j-1:k*j-1+k, col], 'b')

        ax[0].legend(['slow-weights', 'fast-weights'])
        fig.suptitle(optim)
        ax[col].set_title(parameter_formatting(slow_series.name))
        ax[col].set_ylabel('Accuracy')
        ax[col].set_xlabel('fast-weights update')
    plt.show()
