import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'MNIST'
adv_train = True
clean_test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
fgsm_test_path = "{}/adv_fgsm_test_results".format(dataset) if adv_train else "{}/clean_fgsm_test_results".format(dataset)
pgd_test_path = "{}/adv_pgd_test_results".format(dataset) if adv_train else "{}/clean_pgd_test_results".format(dataset)

clean_valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
fgsm_valid_path = "{}/adv_fgsm_valid_results".format(dataset) if adv_train else "{}/clean_fgsm_valid_results".format(dataset)
pgd_valid_path = "{}/adv_pgd_valid_results".format(dataset) if adv_train else "{}/clean_pgd_valid_results".format(dataset)

train_path = "{}/adv_train_results".format(dataset) if adv_train else "{}/clean_train_results".format(dataset)

top_settings = pd.DataFrame()
results= pd.DataFrame()
barplots = False
optims = ['SGD', 'Adam', 'OGDA', 'ExtraSGD', 'ExtraAdam']


def parameter_formatting(index):
    """
    Arguments:
        index(pandas.core.indexes.base.Index): iterable object containing the hyperparameter settings
    Returns:
        list of formatted hyperparameter settings
    """
    return [i.replace('alpha', '\u03B1').replace('steps','k').replace('lr','\u03B3') for i in index]


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


def train_loss_vs_valid_acc():
    """
    Creates 2x2 Plot of validation acc and training loss for optimizers and
    their Lookahead version to allow for comparisons. Top 3 and Bottom 3 settings are plotted
    """

    for optim in optims:
        fig, ax = plt.subplots(2, 2, sharey='all', figsize=(15,7))

        valid_df = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        train_df = pd.read_csv(train_path + "/" + '{}.csv'.format(optim))
        top3_series = valid_df.iloc[-1].sort_values(ascending=False)[:3]  # top 5 settings
        bottom3_series = valid_df.iloc[-1].sort_values(ascending=True)[:3]  # bottom 5 settings

        ax2 = ax[0, 0].twinx()
        two_scales(ax[0, 0], ax2, range(1, valid_df.shape[0] + 1),
                                         valid_df, train_df, top3_series)

        ax2 = ax[1, 0].twinx()
        two_scales(ax[1, 0], ax2, range(1, valid_df.shape[0] + 1),
                                       valid_df, train_df, bottom3_series)

        valid_df_Lookahead = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        train_df_Lookahead = pd.read_csv(train_path + "/" + 'Lookahead-{}.csv'.format(optim))
        top3_LA_series = valid_df_Lookahead.iloc[-1].sort_values(ascending=False)[:3]  # top 5 settings
        bottom3_LA_series = valid_df_Lookahead.iloc[-1].sort_values(ascending=True)[:3]  # bottom 5 settings

        ax2 = ax[0, 1].twinx()
        two_scales(ax[0,1], ax2, range(1, valid_df_Lookahead.shape[0] + 1),
                   valid_df_Lookahead, train_df_Lookahead, top3_LA_series)
        ax2.set_ylabel('Training Loss')
        ax2 = ax[1, 1].twinx()
        two_scales(ax[1, 1], ax2, range(1, valid_df_Lookahead.shape[0] + 1),
                   valid_df_Lookahead, train_df_Lookahead, bottom3_LA_series)
        ax2.set_ylabel('Training Loss')
        ax[0,0].set_title('{}'.format(optim))
        ax[0,1].set_title('Lookahead-{}'.format(optim))
        ax[0,0].set_ylabel('Validation Accuracy')
        ax[1, 0].set_ylabel('Validation Accuracy')
        ax[1,0].set_xlabel('Epochs')
        ax[1,1].set_xlabel('Epochs')
        plt.show()


def get_indices(file):
    col = 1 if 'Lookahead' in file else 0
    if 'ExtraAdam' in file:
        row = 4
    elif 'ExtraSGD' in file:
        row = 3
    elif 'OGDA' in file:
        row = 2
    elif 'Adam' in file:
        row = 1
    elif 'SGD' in file:
        row = 0
    else:
        assert 'Wrong optimizer!'
    return row,col


def lr_aggregation_summaryplot():
    """
    Creates a 5x2 figure of validation accuracies.
    In each row, in the left plot, a validation accuracy plot is created for an optimizer,
    and in the right plot, a valid. acc. plot,
    averaged over all values of la_steps and la_alpha used in the gridsearch,
    is created for the Lookahead version of the optimizer
    """
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    fig, ax = plt.subplots(5, 2, sharey='all')
    fig.set_figheight(20)
    fig.set_figwidth(13)
    fig.suptitle('LR Aggregation')
    for file in os.listdir(clean_valid_path):
        r, c = get_indices(file)
        df = pd.read_csv(clean_valid_path + "/" + file)
        for i, lr in enumerate(learning_rates):

            df2 = pd.DataFrame()
            for col in df.columns:
                if 'lr={}'.format(lr) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            ax[r,c].plot(range(1, df.shape[0] + 1), df2.mean(axis=1),  linestyle='dashed', marker=markers[i], markevery=5)

        ax[r,c].set_title(file.replace('.csv',''))
        ax[r, c].legend(['\u03B3={}'.format(lr) for lr in learning_rates])
    plt.show()


def lr_aggregation_pairplot():
    """Creates a 1x2 figure consisting of two plots.
    On the left: a plot of the validation accuracy for a given optimizer
    On the right: a plot of the valid. acc. of Lookahead version of the optimizer
    averaged over all la_steps and la_alpha used in the gridsearch
    """
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]

    for optim in optims:
        fig, ax = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(5)
        fig.set_figwidth(10)
        df = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df_Lookahead = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, lr in enumerate(learning_rates):
            df2 = pd.DataFrame()
            df2_Lookahead = pd.DataFrame()
            for col in df.columns:
                if 'lr={}'.format(lr) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)
            for col in df_Lookahead.columns:
                if 'lr={}'.format(lr) in col:
                    df2_Lookahead = pd.concat([df2_Lookahead, df_Lookahead[col]], axis=1)

            ax[0].plot(range(1, df.shape[0] + 1), df2.mean(axis=1),  linestyle='dashed', marker=markers[i], markevery=5)
            ax[1].plot(range(1, df.shape[0] + 1), df2_Lookahead.mean(axis=1), linestyle='dashed', marker=markers[i],
                          markevery=5)
        ax[0].set_title('{}'.format(optim))
        ax[1].set_title('Lookahead-{}'.format(optim))
        ax[0].set_ylabel('Accuracy')
        ax[1].set_ylabel('Avg. Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs')
        plt.legend(['\u03B3={}'.format(lr) for lr in learning_rates], loc='lower right')
        plt.show()


def la_steps_aggregation():
    """
    Creates a validation accuracy plot for each Lookahead optimizer
    accuracies for each la_steps parameter averaged over all
    lr and la_alpha values used in the gridsearch
    """
    la_steps_list = [5,10,15]

    for optim in optims:
        df_nolook = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, la_steps in enumerate(la_steps_list):
            df2 = pd.DataFrame()
            for col in df.columns:
                if 'steps={}'.format(la_steps) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            plt.plot(range(1, df.shape[0] + 1), df2.mean(axis=1), marker=markers[i], markevery=5)

        plt.plot(range(1, df_nolook.shape[0]+1), df_nolook[df_nolook.iloc[-1].idxmax()],
                 linestyle='dashed', marker=markers[-1], markevery=5)
        plt.plot(range(1, df_nolook.shape[0]+1), df_nolook[df_nolook.iloc[-1].idxmin()],
                 linestyle='dashed', marker=markers[-2], markevery=5)

        plt.legend(['k={}'.format(la_steps) for la_steps in la_steps_list]+['{}: {}'.format(optim, df_nolook.iloc[-1].idxmax().replace('lr','\u03B3')), '{}: {}'.format(optim, df_nolook.iloc[-1].idxmin().replace('lr','\u03B3'))], loc='lower right')
        plt.title('Lookahead-{}'.format(optim))
        plt.show()


def la_alpha_aggregation():
    """
    Creates a validation accuracy plot for each Lookahead optimizer
    accuracies for each la_alpha parameter averaged over all
    lr and la_steps values used in the gridsearch
    """
    la_alphas = [0.5, 0.75, 0.9]

    for optim in optims:
        df_nolook = pd.read_csv(clean_valid_path + "/" + '{}.csv'.format(optim))
        df = pd.read_csv(clean_valid_path + "/" + 'Lookahead-{}.csv'.format(optim))
        for i, la_alpha in enumerate(la_alphas):
            df2 = pd.DataFrame()
            for col in df.columns:
                if 'alpha={}'.format(la_alpha) in col:
                    df2 = pd.concat([df2, df[col]], axis=1)

            plt.plot(range(1, df.shape[0] + 1), df2.mean(axis=1), marker=markers[i], markevery=5)

        plt.plot(range(1, df_nolook.shape[0] + 1), df_nolook[df_nolook.iloc[-1].idxmax()], linestyle='dashed',
                 marker=markers[-1], markevery=5)
        plt.plot(range(1, df_nolook.shape[0] + 1), df_nolook[df_nolook.iloc[-1].idxmin()], linestyle='dashed',
                 marker=markers[-2], markevery=5)

        plt.legend(['\u03B1={}'.format(la_alpha) for la_alpha in la_alphas]+['{}: {}'.format(optim, df_nolook.iloc[-1].idxmax().replace('lr','\u03B3')), '{}: {}'.format(optim, df_nolook.iloc[-1].idxmin().replace('lr','\u03B3'))], loc='lower right')
        plt.title('Lookahead-{}'.format(optim))
        plt.show()


def valid_acc():
    """Plot of model accuracies validated on clean,fgsm and pgd images of top3 hyperparameter settings"""

    for file in os.listdir(clean_valid_path):
        if not file.startswith('Lookahead'):
            fig, ax = plt.subplots(3, 2, figsize=(12,10))

            df_clean = pd.read_csv(clean_valid_path + "/" + file)
            df_fgsm = pd.read_csv(fgsm_valid_path + "/" + file)
            df_pgd = pd.read_csv(pgd_valid_path + "/" + file)

            df_LA_clean = pd.read_csv(clean_valid_path + "/Lookahead-" + file)
            df_LA_fgsm = pd.read_csv(fgsm_valid_path + "/Lookahead-" + file)
            df_LA_pgd = pd.read_csv(pgd_valid_path + "/Lookahead-" + file)


            # top 3 settings regarding acc on clean images
            top3_clean_valid_series = df_clean.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_clean_valid_series = df_LA_clean.iloc[-1].sort_values(ascending=False)[:3]
            # top 3 settings regarding acc on fgsm perturbed images
            top3_fgsm_valid_series = df_fgsm.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_fgsm_valid_series = df_LA_fgsm.iloc[-1].sort_values(ascending=False)[:3]
            # top 3 settings regarding acc on pgd perturbed images
            top3_pgd_valid_series = df_pgd.iloc[-1].sort_values(ascending=False)[:3]
            top3_LA_pgd_valid_series = df_LA_pgd.iloc[-1].sort_values(ascending=False)[:3]

            for i, col in enumerate(top3_clean_valid_series.index):
                ax[0,0].plot(range(1, df_clean.shape[0] + 1), df_clean[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_clean_valid_series.index):
                ax[0, 1].plot(range(1, df_LA_clean.shape[0] + 1), df_LA_clean[col], marker=markers[i], markevery=5)

            for i, col in enumerate(top3_fgsm_valid_series.index):
                ax[1,0].plot(range(1, df_fgsm.shape[0] + 1), df_fgsm[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_fgsm_valid_series.index):
                ax[1, 1].plot(range(1, df_LA_fgsm.shape[0] + 1), df_LA_fgsm[col], marker=markers[i], markevery=5)

            for i, col in enumerate(top3_pgd_valid_series.index):
                ax[2,0].plot(range(1, df_pgd.shape[0] + 1), df_pgd[col], marker=markers[i], markevery=5)
            for i, col in enumerate(top3_LA_pgd_valid_series.index):
                ax[2,1].plot(range(1, df_LA_pgd.shape[0] + 1), df_LA_pgd[col], marker=markers[i], markevery=5)

            for i in range(3):
                for j in range(2):
                    ax[i,j].set_ylim([0,1.1])

            legend_loc = 'upper left' if dataset == 'CIFAR10' else 'lower right'
            ax[0,0].legend(parameter_formatting(top3_clean_valid_series.index), loc=legend_loc)
            ax[0,1].legend(parameter_formatting(top3_LA_clean_valid_series.index), loc=legend_loc)
            ax[1,0].legend(parameter_formatting(top3_fgsm_valid_series.index), loc=legend_loc)
            ax[1,1].legend(parameter_formatting(top3_LA_fgsm_valid_series.index), loc=legend_loc)
            ax[2,0].legend(parameter_formatting(top3_pgd_valid_series.index), loc=legend_loc)
            ax[2,1].legend(parameter_formatting(top3_LA_pgd_valid_series.index), loc=legend_loc)

            ax[2,0].set_xlabel('Epochs')
            ax[2, 1].set_xlabel('Epochs')
            ax[0, 0].set_ylabel('Clean Accuracy')
            ax[1, 0].set_ylabel('FGSM Accuracy')
            ax[2, 0].set_ylabel('PGD Accuracy')
            ax[0,0].set_title('{}'.format(file.replace('.csv', '')))
            ax[0,1].set_title('Lookahead-{}'.format(file.replace('.csv', '')))
            plt.show()


#la_alpha_aggregation()
#la_steps_aggregation()
#lr_aggregation_pairplot()
#lr_aggregation_summaryplot()
#top5_plots()
#train_loss_vs_valid_acc()
valid_acc()
