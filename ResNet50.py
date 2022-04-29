import torch
import torchvision
import torchvision.transforms as transforms
import Lookahead_tutorial
import project_utils
from easydict import EasyDict
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from absl import app, flags
import time

"""
1) https://pytorch.org/hub/pytorch_vision_resnet/
2) https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)
"""

FLAGS = flags.FLAGS

transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  # convert PIL image into tensor and transform to match ResNet50 requirements (see 2))


def cifar10_training(train_set, net, optimizer, loss_fn, device, adv_train=False):
    for epoch in range(1, FLAGS.nb_epochs + 1):
        start_t = time.time()
        train_loss = 0.0
        for x, y in train_set:  # take batches of batch_size many inputs stored in x and targets stored in y
            x, y = x.to(device), y.to(device)
            if adv_train:
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()  # explained in 3). Sets the gradient to zero
            loss = loss_fn(net(x), y)  # creates a new loss_fn (torch.nn.crossentropyloss) class instance
            loss.backward()  # computes the gradient - see also 4)
            optimizer.step()  # updates the parameters - see also 4)
            train_loss += loss.item()  # extracts loss value
        print(
            "epoch: {}/{}, train loss: {:.3f} computed in {:.3f} seconds".format(
                epoch, FLAGS.nb_epochs, train_loss, time.time()-start_t
            )
        )


def cifar10_evaluation(test_set, net, device):
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_set:
        x, y = x.to(device), y.to(device)

        _, y_pred = net(x).max(1)  # model prediction on clean examples

        report.nb_test += y.size(0)  # counts how many examples are in the batch
        report.correct += y_pred.eq(y).sum().item()  # counts how many examples in the batch are predicted correctly

        if FLAGS.fgsm_att:
            x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
            _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples

            report.correct_fgm += y_pred_fgm.eq(y).sum().item()  # counts correctly predicted fgsm examples

        if FLAGS.pgd_att:
            x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples

            report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # counts correctly predicted pgd examples

    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )

    if FLAGS.fgsm_att is True:
        print(
            "test acc on FGM adversarial examples (%): {:.3f}".format(
                report.correct_fgm / report.nb_test * 100.0
            )
        )
    if FLAGS.pgd_att is True:
        print(
            "test acc on PGD adversarial examples (%): {:.3f}".format(
                report.correct_pgd / report.nb_test * 100.0
            )
        )


def main(_):
    data = project_utils.dataset_spec(transform, FLAGS.batch_size)

    net = torchvision.models.resnet50()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")  # averages over all losses

    optimizer = Lookahead_tutorial.Lookahead(torch.optim.Adam(net.parameters(), lr=1e-3), la_alpha=0.5)
    # or: torch.optim.Adam(net.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if gpu is available
    if device == "cuda":
        net = net.cuda()  # transfers to gpu

    # Train ResNet50 model
    net.train()
    cifar10_training(data.train, net, optimizer, loss_fn, device)

    # Evaluation
    net.eval()
    cifar10_evaluation(data.test, net, device)


def settings(nb_epochs=1, batch_size=50, eps=0.3,adv_train=True, fgsm_att=False, pgd_att=False):
    flags.DEFINE_integer("nb_epochs", nb_epochs, "Number of epochs.")
    flags.DEFINE_integer("batch_size", batch_size, "Size of Minibatch")
    flags.DEFINE_float("eps", eps, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", adv_train, "Use adversarial training (on PGD adversarial examples)."
    )
    flags.DEFINE_bool(
        "fgsm_att", fgsm_att, "Use FGSM attack in evaluation"
    )
    flags.DEFINE_bool(
        "pgd_att", pgd_att, "Use PGD attack in evaluation"
    )


if __name__ == "__main__":

    if project_utils.yes_no_check('Run on standard settings?'):
        settings()
    else:
        settings(
            nb_epochs=project_utils.int_query('Number of epochs'),
            batch_size=project_utils.int_query('Batch size'),
            adv_train=project_utils.yes_no_check('Adversarial Training?'),
            fgsm_att=project_utils.yes_no_check('FGSM Attack during testing?'),
            pgd_att=project_utils.yes_no_check('PGD Attack during testing')
            )

    app.run(main)

