import pandas as pd
import matplotlib.pyplot as plt

filename = "data/MNIST-clean_results.csv"
df = pd.read_csv(filename)

markers=('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')

for i, col in enumerate(df.columns):
    plt.plot(range(1,df.shape[0]+1), df[col], linestyle='dashed', marker=markers[i], markevery=10)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(df.columns)
plt.show()

