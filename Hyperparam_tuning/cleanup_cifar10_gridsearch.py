import pandas as pd
import os
import numpy as np

dataset = 'CIFAR10'
adv_train = True
test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
train_path = "{}/adv_train_results".format(dataset) if adv_train else "{}/clean_train_results".format(dataset)
optims = ['SGD', 'Adam', 'OGDA', 'ExtraSGD', 'ExtraAdam']

for file in os.listdir(test_path):
    df = pd.read_csv(test_path + "/" + file)
    if file.startswith('Lookahead'):
        if file.startswith('Lookahead-Adam'):
            'Do nothing'
        'Do'
    else:
        df = df[5:]

    break