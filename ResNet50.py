import torch
import torchvision
from Optimizer import extragradient, Lookahead, OGDA
import project_utils
import data_utils
from absl import app

"""
1) https://pytorch.org/hub/pytorch_vision_resnet/
2) https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)
"""


def main(_):
    data = data_utils.ld_dataset()

    net = torchvision.models.resnet50()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)  # transfers to gpu if available

    optimizer_name = project_utils.query_optim()
    if optimizer_name == 'LA-SGD':
        optimizer = Lookahead.Lookahead(torch.optim.SGD(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-Adam':
        optimizer = Lookahead.Lookahead(torch.optim.Adam(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-ExtraSGD':
        optimizer = Lookahead.Lookahead(extragradient.ExtraSGD(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-ExtraAdam':
        optimizer = Lookahead.Lookahead(extragradient.ExtraAdam(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-OGDA':
        optimizer = Lookahead.Lookahead(OGDA.OGDA(net.parameters(), lr=1e-3))
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    elif optimizer_name == 'ExtraSGD':
        optimizer = extragradient.ExtraSGD(net.parameters(), lr=1e-3)
    elif optimizer_name == 'ExtraAdam':
        optimizer = extragradient.ExtraAdam(net.parameters(), lr=1e-3)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    elif optimizer_name == 'OGDA':
        optimizer = OGDA.OGDA(net.parameters(), lr=1e-3)
    else:
        raise 'Wrong optimizer'

    # Train model
    net.train()
    data_utils.my_training(data.train, net, optimizer, device)

    # Evaluation
    net.eval()
    data_utils.my_evaluation(data.test, net, device)


if __name__ == "__main__":

    if project_utils.yes_no_check('Run on standard settings?'):
        data_utils.settings(dataset=project_utils.query_dataset())
    else:
        data_utils.settings(
            dataset=project_utils.query_dataset(),
            nb_epochs=project_utils.int_query('Number of epochs'),
            adv_train=project_utils.yes_no_check('Adversarial Training?'),
            fgsm_att=project_utils.yes_no_check('FGSM Attack during testing?'),
            pgd_att=project_utils.yes_no_check('PGD Attack during testing')
            )

    app.run(main)

