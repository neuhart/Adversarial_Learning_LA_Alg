import numpy as np
import csv
import matplotlib.pyplot as plt

"Data Import"
"Train Data"
trainname = 'train.csv'
Attributes_X=[]
Labels=[]
with open(trainname, newline='') as csvfile: #importing training data
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Attributes_X.append(row[1:]) #Importing the Input Data
        Labels.append(row[0]) #Input the Output Data (Labels, 0-9)
Attributes_X=[np.array(list(map(int, x))) for x in Attributes_X[1:]] #Converting from string into int, removing first row (header), converting into array
Labels=np.array(list(map(int, Labels[1:])))

"Test Data"
testname = 'train.csv'
t_Attributes_X=[]
t_Labels=[]
with open(testname, newline='') as csvfile: #importing test data
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        t_Attributes_X.append(row[1:]) #Importing the Input Data
        t_Labels.append(row[0]) #Input the Output Data (Labels, 0-9)
t_Attributes_X=[np.array(list(map(int, x))) for x in t_Attributes_X[1:]] #Converting from string into int, removing first row (header), converting into array
t_Labels=np.array(list(map(int, t_Labels[1:])))

