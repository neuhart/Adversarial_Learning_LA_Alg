import pandas as pd
import matplotlib.pyplot as plt

filename = "data/results.csv"

df = pd.read_csv(filename)

for col in df.columns:
    plt.plot(range(df.shape[0]), df[col])
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(df.columns)
plt.show()

