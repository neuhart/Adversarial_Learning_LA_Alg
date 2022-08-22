import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import numpy as np
from Utils import project_utils
from easydict import EasyDict
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.utils import optimize_linear

X = '0-1 box'
Theta = '0-1 box'




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
    def __init__(self, x1, x2):
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

    def clear(self):
        self.x1= []
        self.x2= []


def Loss_fn(x):
    return x[0]**2 + 5*x[1]**2


def nLoss_fn(x):
    return -(x[0]**2 + 5*x[1]**2)

def plot_loss_contour():
    xlist = np.linspace(-1.0, 1.0, 100)
    ylist = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = Loss_fn((X, Y))
    ax.contour(X, Y, Z)


def pga(x):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    loss = Loss_fn(model(x))
    loss.backward()
    adv_x = x + optimize_linear(x.grad,0.5)
    a = adv_x.clone().detach().numpy()
    ax.plot(a[0], a[1], marker='x')
    print(a)
    return adv_x


def training(nb_epochs):
    x_in = torch.Tensor(inputs.get_single_weight(0))
    ax.plot(x_in[0],x_in[1], marker='x')


    for epoch in range(nb_epochs):
        adv_x = pga(x_in)
        optimizer.zero_grad()
        path.add_weight(model(adv_x).detach().numpy())
        loss = Loss_fn(model(adv_x))
        loss.backward()
        optimizer.step()

random.seed(10)
weights = Weights([1 - 2 * random.random() for i in range(3)], [1 - 2 * random.random() for i in range(3)])
#inputs = Weights([(1 - 2 * random.random())/10 for i in range(3)], [(1 - 2 * random.random())/10 for i in range(3)])
inputs = Weights([-0.8,1,1],[0.8,1,1])
optim_list = ['SGD','Lookahead-SGD', 'Adam']

fig, ax = plt.subplots()
plot_loss_contour()
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
for iterate, optim in enumerate(optim_list):
    torch.manual_seed(0)
    model = NN()
    path = Weights([],[])
    optimizer = project_utils.set_optim(EasyDict(adv_train=True, dataset='CIFAR10'), optim, model)
    training(nb_epochs=100)

    colours = ('g', 'r','c', 'y', 'm', 'k')
    ax.scatter(path.x1[::10], path.x2[::10], color=colours[iterate])
ax.legend(optim_list)
plt.show()


"""
fig, ax = plt.subplots(1,2)
ax[0] = plt.scatter(weights.x1, weights.x2)
ax[1] = plt.scatter(inputs.x1, inputs.x2)
plt.show()
"""

