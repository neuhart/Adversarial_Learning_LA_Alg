import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'MNIST'
adv_train = False
top_settings = pd.DataFrame()
results= pd.DataFrame()
optims = ['Lookahead-SGD', 'Lookahead-Adam', 'Lookahead-OGDA', 'Lookahead-ExtraSGD', 'Lookahead-ExtraAdam']

optim='Lookahead-SGD'
fast_weights_valid_path = "{}/adv_valid_results/{}_fast-weights.csv".format(dataset, optim) if adv_train else "{}/clean_valid_results/{}_fast-weights.csv".format(dataset, optim)
slow_weights_valid_path = "{}/adv_valid_results/{}_slow-weights.csv".format(dataset, optim) if adv_train else "{}/clean_valid_results/{}_slow-weights.csv".format(dataset, optim)

df1 = pd.read_csv(fast_weights_valid_path)
df2= pd.read_csv(slow_weights_valid_path)

df3 = df2.iloc[:,0]
df3 = df3[df3.index % 5 == 0]
df4 = df1[df1.index % 5 == 0]



for i in range(3):
    for j in range(10):
        plt.plot(range(j,j+4),df1.iloc[j:j+4,i])
    plt.plot(range(1,10+1),df3[:10], linestyle='dashed')
    plt.legend(['fast-weights','slow-weights'])
    plt.show()