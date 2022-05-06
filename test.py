import Lookahead_tutorial
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from torch.utils.data import Dataset
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import extragradient

"""
This is a simple example of neural networks used in the regression setting, in this case for approximating the sine
function. Two networks are being trained, although both have the same architecture. The first one uses ADAM and the 
second the Lookahead optimizer
"""

batch_size = 50


class Sin_Dataset(Dataset):
    # Create own dataset (subclass) in pytorch

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]




# Create toy dataset
x_values = torch.from_numpy(np.arange(-5, 5, 0.02).astype(np.float32))
y_values = np.sin(x_values)

trainset = Sin_Dataset(x_values, y_values)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# if num_workers>0 is necessary, trainset needs to be converted into list before that!

x_t_values = torch.from_numpy(np.arange(-4.9, 5.1, 0.02).astype(np.float32))  # float required
y_t_values = np.sin(x_t_values)

testset = Sin_Dataset(x_t_values, y_t_values)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


data = EasyDict(train=trainloader, test=testloader)


# specify network architecture
class ffNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self):  # method thats called when a new NN instance is created
        super(ffNN, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = x.view(-1, 1)  # necessary to get correct shape for first layer
        x = torch.sigmoid(self.fc1(x))
        return x


def evaluation_fn(test_data, device, net_model):
    test_loss = 0.0
    for x, y in test_data:
        x, y = x.to(device), y.to(device)
        y_pred = net_model(x)
        y = y.view(-1, 1)  # reshapes y to match shape of x (to avoid error message in loss)

        loss = loss_fn(y_pred, y)
        test_loss += loss.item()

    print(
        "MSE (%): {:.3f}".format(
            test_loss
        )
    )

if __name__ == '__main__':
    net_model = ffNN()
    optimizer=extragradient.ExtraAdam(net_model.parameters(), lr=1e-2)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if gpu is available
    if device == "cuda":
        net_model = net_model.cuda()  # transfers to gpu
    loss_fn = torch.nn.MSELoss()  # averages over all losses

    nb_epochs = 50

    for p in optimizer.param_groups:
        print(p)
    net_model.eval()
    evaluation_fn(data.test, device, net_model)

    # Train vanilla model
    net_model.train()
    for epoch in range(1, nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:  # take batches of batch_size many inputs stored in x and targets stored in y
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # Sets the gradient to zero
            y = y.view(-1,1)  # reshapes y to match shape of x (to avoid error message in loss)
            loss = loss_fn(input=net_model(x), target=y)  # forward pass
            loss.backward()  # backward pass
            optimizer.extrapolation()
            optimizer.zero_grad()
            loss = loss_fn(input=net_model(x), target=y)  # forward pass
            loss.backward()  # backward pass
            optimizer.step()
            train_loss += loss.item()  # extracts loss value

        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, nb_epochs, train_loss))

    for p in optimizer.param_groups:
        print(p)

    # test and plot
    net_model.eval()
    evaluation_fn(data.test, device, net_model)








