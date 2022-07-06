import pandas as pd
import os
import numpy as np

dataset = 'CIFAR10'
adv_train = True
test_path = "{}/adv_test_results".format(dataset) if adv_train else "{}/clean_test_results".format(dataset)
valid_path = "{}/adv_valid_results".format(dataset) if adv_train else "{}/clean_valid_results".format(dataset)
train_path = "{}/adv_train_results".format(dataset) if adv_train else "{}/clean_train_results".format(dataset)
optims = ['SGD', 'Adam', 'OGDA', 'ExtraSGD', 'ExtraAdam']


def remove_duplicate_1(file, path):
    df = pd.read_csv(path + "/" + file)

    if not file.startswith('Lookahead-Adam'):
        'Do nothing'
    else:
        for col in df.columns:
            if col.endswith('.1'):
                df.rename(columns={col: col[:-2]}, inplace=True)
        df.to_csv(path + "/" + file, index=False)


def clean_test(file):
    df = pd.read_csv(test_path + "/" + file)
    if not file.startswith('Lookahead-Adam'):
        'Do nothing'
    else:
        dfnew = pd.DataFrame(df.iloc[1]).transpose()
        dfold = pd.DataFrame(df.iloc[0]).transpose()
        dfnew.to_csv(test_path + '/' + file, index=False)
        dfold.to_csv(
            "C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup/adv_test_results/" + file,
            index=False)
    remove_duplicate_1(file, test_path)

def clean_train(file):
    df = pd.read_csv(train_path + "/" + file)
    if file.startswith('Lookahead'):
        if not file.startswith('Lookahead-Adam'):
            'Do nothing'
        else:
            dfnew = df.iloc[:, 45:]
            dfold = df.iloc[:, :45]
            dfnew.to_csv(train_path + '/' + file, index=False)
            dfold.to_csv("C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup/adv_train_results/" + file, index=False)
    else:

        dfnew = df.iloc[:, 5:]
        dfold = df.iloc[:, :5]
        dfnew.to_csv(train_path + '/' + file, index=False)
        dfold.to_csv(
            "C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup/adv_train_results/" + file,
            index=False)

    remove_duplicate_1(file, train_path)

def clean_valid(file):
    df = pd.read_csv(valid_path + "/" + file)
    if file.startswith('Lookahead'):
        if not file.startswith('Lookahead-Adam'):
            'Do nothing'
        else:
            dfnew = df.iloc[:, 45:]
            dfold = df.iloc[:, :45]
            dfnew.to_csv(valid_path + '/' + file, index=False)
            dfold.to_csv(
                "C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup/adv_valid_results/" + file,
                index=False)
    else:
    
        dfnew = df.iloc[:, 5:]
        dfold = df.iloc[:, :5]
        dfnew.to_csv(valid_path + '/' + file, index=False)
        dfold.to_csv("C:/Users/phili/OneDrive/Desktop/Uni_Philip/Master Mathematik/Masterarbeit/GitHub/CIFAR10_gridsearch_smallnetwork_backup/adv_valid_results/" + file, index=False)

    remove_duplicate_1(file, valid_path)

"""
for file in os.listdir(test_path):
    clean_test(file)
    clean_train(file)
    clean_valid(file)
"""
remove_duplicate_1('Lookahead-Adam.csv',test_path)
remove_duplicate_1('Lookahead-Adam.csv',train_path)
remove_duplicate_1('Lookahead-Adam.csv',valid_path)
