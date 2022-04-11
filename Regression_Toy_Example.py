import Lookahead
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from torch.utils.data import Dataset

class sin_Dataset(Dataset):
    """Create own dataset (subclass) in pytorch"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


batch_size = 5
# Load sin dataset
x_values = torch.from_numpy(np.arange(0,5,0.2).astype(np.float32))
y_values = np.sin(x_values)

trainset = sin_Dataset(x_values,y_values)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

x_t_values = torch.from_numpy(np.arange(0.1, 5.1, 0.2).astype(np.float32))  # float required
y_t_values = np.sin(x_values)

testset = sin_Dataset(x_t_values,y_t_values)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

data = EasyDict(train=trainloader, test=testloader)

class ffNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self): #method thats called when a new NN instance is created
        super(ffNN, self).__init__() #means to call a bound __init__ from the parent class that follows SomeBaseClass's
        # child class (the one that defines this method) in the instance's Method Resolution Order (MRO)
        # in this case calls __init__ of torch.nn.Module
        self.fc1 = nn.Linear(1,20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40,1)

    def forward(self, x):
        x = x.view(-1,1)  # necessary to get correct shape for first layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.flatten(x)  # reshape x to match shape of y
        return x



if __name__ == '__main__':
    nb_epochs = 20
    net = ffNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if gpu is available
    if device == "cuda":
        net = net.cuda()  # transfers to gpu
    loss_fn = torch.nn.MSELoss()   # averages over all losses
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)

    # Train vanilla model
    net.train()
    for epoch in range(1, nb_epochs+1):
        train_loss = 0.0
        for x, y in data.train:  # take batches of batch_size many inputs stored in x and targets stored in y
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # Sets the gradient to zero
            loss = loss_fn(net(x), y) #need to adapt shape of y to shape of x to avoid errors
            loss.backward()  # computes the gradient - see also 4)
            optimizer.step()  # updates the parameters - see also 4)
            train_loss += loss.item()  # extracts loss value
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, nb_epochs, train_loss))

    # test and plot
    preds=[]
    report = EasyDict(nb_test=0, correct=0)
    test_loss = 0.0
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        y_pred = net(x)

        loss = loss_fn(net(x), y)
        test_loss += loss.item()

        preds.append(y_pred)

    Y_pred = torch.cat(preds, dim=0) #concatenate batch predictions

    print(
        "MSE (%): {:.3f}".format(
            test_loss
        )
    )


    plt.plot(x_values, y_values)
    plt.plot(x_t_values,y_t_values)
    plt.plot(x_t_values, Y_pred.detach().numpy())
    plt.title('Neural Network Regression Toy Ex.')
    plt.legend(['training', 'testing', 'prediction'])
    plt.show()