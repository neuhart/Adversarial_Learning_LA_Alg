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


for optim in ['Lookahead-SGD','Lookahead-Adam','Lookahead-OGDA', 'Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']:
    if adv_train:
        fast_weights_valid_path = "{}/adv_valid_results/{}-fast-weights.csv".format(dataset, optim)
        slow_weights_valid_path = "{}/adv_valid_results/{}-slow-weights.csv".format(dataset, optim)
    else:
        fast_weights_valid_path = "{}/clean_valid_results/{}_fast-weights.csv".format(dataset, optim)
        slow_weights_valid_path = "{}/clean_valid_results/{}_slow-weights.csv".format(dataset, optim)

    df1 = pd.read_csv(fast_weights_valid_path)
    df2= pd.read_csv(slow_weights_valid_path)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(15,5))

    for col in range(df2.shape[1]):
        df3 = df2.iloc[:,col]

        # slow weights
        ax[col].plot([4*i+4 for i in range(10)],  df3[(df3.index + 1) % 5 == 0][:10], 'r',  linestyle='dashed', marker='x')

        # fast weights
        ax[col].plot(range(1,5), df1.iloc[0:4, col], 'b')
        for j in range(1,10):
            ax[col].plot(range(j*4,4*j+5), df1.iloc[5*j-1:5*j-1+5, col], 'b')

        ax[0].legend(['fast-weights', 'slow-weights'])
        fig.suptitle(optim)
        ax[col].set_title(parameter_formatting(df3.name))
        ax[col].set_ylabel('Accuracy')
        ax[col].set_xlabel('fast-weights update')
    plt.show()
