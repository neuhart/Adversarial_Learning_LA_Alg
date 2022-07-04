import pandas as pd
import os
import numpy as np

dataset = 'CIFAR10'
adv_train = True
test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
train_path = "{}/adv_train_results".format(dataset) if adv_train else "{}/clean_train_results".format(dataset)
optims = ['SGD', 'Adam', 'OGDA', 'ExtraSGD', 'ExtraAdam']

for file in os.listdir(train_path):
    df = pd.read_csv(train_path + "/" + file)
    if file.startswith('Lookahead'):
        if file.startswith('Lookahead-Adam'):
            'Do nothing'
        else:
            dfnew = df.iloc[:, 45:]
            dfold = df.iloc[:, :45]
            dfnew.to_csv(train_path + '/' + file, index=False)
            dfold.to_csv("C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup" + '/' + file, index=False)
    else:
        dfnew = df.iloc[:, 5:]
        dfold = df.iloc[:, :5]
        dfnew.to_csv(train_path + '/' + file, index=False)
        dfold.to_csv("C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup" + '/' + file, index=False)
    break