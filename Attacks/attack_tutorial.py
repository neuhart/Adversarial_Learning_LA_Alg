from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import project_utils
import time

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

"""
1) https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
2) https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
3) https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
4) https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
5) https://abseil.io/docs/python/guides/flags
"""

FLAGS = flags.FLAGS  # see 5) defines globals variables


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):  # method thats called when a new CNN instance is created
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        # in_channels = 1 = number of input stacks (e.g, colours),
        # out_channels= 64 = number of output stacks/number of filters in the layer ,
        # kernel_size= 8 = size of each convolution filter (8x8),
        # stride=1 (shift of kernel/filter from left to right, top to bottom when convolution is performed)
        # in this case: convoluted image with width W_after= [W_before-(kernel size-1)]/stride and
        # height H_after = [H_before-(kernel size-1)]/stride , else see 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, 10)  # image dimensions 3x3 with 128 channels (128,3,3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)  # reshapes tensor from (batch_size, #channels, height, width) to
        # (batch_size, 128*3*3); (image dimensions 3x3 with 128 channels (128,3,3))
        x = self.fc(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor()
     ])


def main(_):
    # Load training and test data
    data = project_utils.ld_cifar10(transform, batch_size=5)
    # Instantiate model, loss, and optimizer for training
    net = CNN(in_channels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu" # check if gpu is available
    if device == "cuda":
        net = net.cuda()  # transfers to gpu
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean") # averages over all losses
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    start_time = time.time()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        epoch_start_time = time.time()
        train_loss = 0.0
        for x, y in data.train:  # take batches of batch_size many inputs stored in x and targets stored in y
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()  # explained in 3). Sets the gradient to zero
            loss = loss_fn(net(x), y)  # creates a new loss_fn (torch.nn.crossentropyloss) class instance
            loss.backward()  # computes the gradient - see also 4)
            optimizer.step()  # updates the parameters - see also 4)
            train_loss += loss.item()  # extracts loss value
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )
        print(
            "epoch {}/{}, runtime: {:.3f}".format(
                epoch, FLAGS.nb_epochs, time.time() - epoch_start_time
            )
        )
    print(
        "training time: {:.3f}".format(
            time.time()-start_time
        )
    )
    start_test_time = time.time()

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in data.test:
        start_test_ex = time.time()
        x, y = x.to(device), y.to(device)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples, net(x) returns a (batch_size, #classes) tensor
        # with a 10-dimensional class probability vector in the second dim, .max(1) takes maximum value of prob. vector
        _, y_pred_fgm = net(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = net(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)  # counts how many examples are in the batch
        report.correct += y_pred.eq(y).sum().item()  # counts how many examples in the batch are predicted correctly
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


if __name__ == "__main__":  # only runs when this file/module is the main module (module that you run)
                            # doesnt run if this file/module is called from or imported from another module
    flags.DEFINE_integer("nb_epochs", 1, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)  # runs main function