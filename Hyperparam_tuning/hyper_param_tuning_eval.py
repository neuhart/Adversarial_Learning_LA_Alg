import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'FashionMNIST'
test_path = "{}/adv_test_results".format(dataset)
top_settings = pd.DataFrame()
results= pd.DataFrame()
show_plots = False


for file in os.listdir(test_path):
    df = pd.read_csv(test_path + "/" + file)
    if show_plots:
        fig, ax = plt.subplots(1, 2, sharey='all')
        sns.barplot(ax=ax[0],x=df.iloc[0].sort_values(ascending=False)[:5].index, y=df.iloc[0].sort_values(ascending=False)[:5])
        sns.barplot(ax=ax[1],x=df.iloc[0].sort_values(ascending=True)[:5].index, y=df.iloc[0].sort_values(ascending=True)[:5])
        ax[0].set_ylabel('Accuracy')
        ax[1].set_ylabel('')
        # plt.xticks(rotation=90) only rotated xticks of right-hand plot
        # Workaround:
        ax[0].set_xticklabels(df.iloc[0].sort_values(ascending=False)[:5].index,rotation=90)
        ax[1].set_xticklabels(df.iloc[0].sort_values(ascending=True)[:5].index, rotation=90)

        ax[0].set_title('Top 5')
        ax[1].set_title('Bottom 5')
        fig.suptitle('{}'.format(file.replace('.csv', '')))
        plt.tight_layout()
        plt.show()






