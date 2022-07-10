import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

dataset = 'CIFAR10'
adv_train=True
optim='Lookahead-SGD'
fast_weights_valid_path = "{}/adv_valid_results/{}-fast-weights.csv".format(dataset, optim) if adv_train else "{}/clean_valid_results/{}_fast-weights.csv".format(dataset, optim)
slow_weights_valid_path = "{}/adv_valid_results/{}-slow-weights.csv".format(dataset, optim) if adv_train else "{}/clean_valid_results/{}_slow-weights.csv".format(dataset, optim)

df1 = pd.read_csv(fast_weights_valid_path)
df2= pd.read_csv(slow_weights_valid_path)

df3 = df2.iloc[:,2]
df4 = df1.iloc[:,2]
df3 = df3[(df3.index + 1) % 5 == 0]
df4 = df4[(df4.index+ 1) % 5 == 0]
for j in range(10):
    plt.plot(range(j*5,5*j+5), df1.iloc[5*j:5*j+5, 2])
plt.plot([5*i+5 for i in range(10)], df3[:10], linestyle='dashed', marker='x')
plt.legend(['fast-weights', 'slow-weights'])
plt.title(df3.name)
plt.show()
