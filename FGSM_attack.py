from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

"""
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

FLAGS = flags.FLAGS


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1): #method thats called when a new CNN instance is created
        super(CNN, self).__init__() #means to call a bound __init__ from the parent class that follows SomeBaseClass's
        # child class (the one that defines this method) in the instance's Method Resolution Order (MRO)
        # in this case calls __init__ of torch.nn.Module
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        # in_channels = 1 = number of input stacks (e.g, colours),
        # out_channels= 64 = number of output stacks/number of filters in the layer ,
        # kernel_size= 8 = size of each convolution filter (8x8),
        # stride=1 (shift of kernel/filter from left to right, top to bottom when convolution is performed)
        # convoluted image with width W_after= [W_before-(kernel size-1)]/stride and
        # height H_after = [H_before-(kernel size-1)]/stride
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, 10) #image dimensions 3x3 with 128 channels (128,3,3)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.conv1(x))
        print(x.size())
        x = F.relu(self.conv2(x))
        print(x.size())
        x = F.relu(self.conv3(x))
        print(x.size())
        x = x.view(-1, 128 * 3 * 3) #image dimensions 3x3 with 128 channels (128,3,3)
        print(x.size())
        x = self.fc(x)
        return x


def ld_cifar10():
    """Load training and test data."""
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )# convert PIL image into tensor

    batch_size = 4  # number of samples per batch

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)  # load in training set

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # download test set, store into ./data and apply transform

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)  # load in test set

    return EasyDict(train=trainloader, test=testloader)


def main(_):
    # Load training and test data
    data = ld_cifar10()

    # Instantiate model, loss, and optimizer for training
    net = CNN(in_channels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu" #check if gpu is available
    if device == "cuda":
        net = net.cuda() #transfers to gpu
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean") #averages over all losses
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) 

    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = net(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = net(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )


if __name__ == "__main__": #only runs when this file/module is the main module (module that you run)
                            #doesnt run if this file/module is called from or imported from another module
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)