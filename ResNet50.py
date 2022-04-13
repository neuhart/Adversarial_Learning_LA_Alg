from easydict import EasyDict
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time

"""
https://pytorch.org/hub/pytorch_vision_resnet/
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)
"""

batch_size = 50  # number of samples per batch
nb_epochs = 5


def ld_cifar10():
    """Load training and test data."""
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  # convert PIL image into tensor


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # download training set, store into ./data and apply transform

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)  # load in training set

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # download test set, store into ./data and apply transform

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)  # load in test set

    return EasyDict(train=trainloader, test=testloader)

def main():
    # Load training and test data
    data = ld_cifar10()
    # Instantiate model, loss, and optimizer for training
    net = torchvision.models.resnet50()
    device = "cuda" if torch.cuda.is_available() else "cpu" # check if gpu is available
    if device == "cuda":
        net = net.cuda() #transfers to gpu
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean") # averages over all losses
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(1, nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:  # take batches of batch_size many inputs stored in x and targets stored in y
            start_time = time.time()
            print('next example')
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # explained in 3). Sets the gradient to zero
            loss = loss_fn(net(x), y)  # creates a new loss_fn (torch.nn.crossentropyloss) class instance
            f_time = time.time()
            print('forward pass completed in {:.3f}'.format(f_time-start_time))
            loss.backward()  # computes the gradient - see also 4)
            print('backward pass completed in {:.3f}'.format(time.time()-f_time))
            optimizer.step()  # updates the parameters - see also 4)
            train_loss += loss.item()  # extracts loss value
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, nb_epochs, train_loss
            )
        )

if __name__  == "__main__":
    main()