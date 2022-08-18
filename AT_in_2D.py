import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

X = '0-1 box'
Theta = '0-1 box'

random.seed(10)


class NN(torch.nn.Module):
    def __init__(self):  # method thats called when a new NN instance is created
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(1, 2)

    def forward(self, x):
        # x = x.view(-1, 1)  # necessary to get correct shape for first layer
        x = self.fc2(self.fc1(x))
        return x


class Weights:
    def __init__(self, x1=[], x2=[]):
        """
        Arguments:
            x1 (list): List containing first coordinate of each weight
            x2 (list): list containing second coordinate of each weight
        """
        self.x1 = x1  # first coordinate
        self.x2 = x2  # second coordinate
        assert len(x1) == len(x2), 'Number of first coordinates does not match number of second coordinates'

    def get_single_weight(self, index):  # return single weight
        return self.x1[index], self.x2[index]

    def add_weight(self,weight):
        self.x1.append(weight[0])
        self.x2.append(weight[1])


def Loss_fn(x):
    return x[0]**2 + x[1]**2


def plot_prediction_path(path):
    plt.scatter(path.x1,path.x2)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()


def training(nb_epochs):
    x_in = torch.Tensor(inputs.get_single_weight(0))
    path = Weights()
    for epoch in range(nb_epochs):
        optim.zero_grad()
        path.add_weight(model(x_in).detach().numpy())
        loss = Loss_fn(model(x_in))
        loss.backward()
        optim.step()
    plot_prediction_path(path)


weights = Weights([1 - 2*random.random() for i in range(3)], [1 - 2*random.random() for i in range(3)])
inputs = Weights([1 - 2*random.random() for i in range(3)], [1 - 2*random.random() for i in range(3)])


model = NN()
optim = torch.optim.SGD(model.parameters(), lr=1e-2)
training(nb_epochs=100)

"""
fig, ax = plt.subplots(1,2)
ax[0] = plt.scatter(weights.x1, weights.x2)
ax[1] = plt.scatter(inputs.x1, inputs.x2)
plt.show()
"""

