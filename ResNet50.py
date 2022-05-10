import torch
import torchvision
import torchvision.transforms as transforms
import Lookahead_tutorial
import OGDA
import extragradient
import project_utils
import data_utils
from absl import app

"""
1) https://pytorch.org/hub/pytorch_vision_resnet/
2) https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)
"""

transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  # convert PIL image into tensor and transform to match ResNet50 requirements (see 2))


def main(_):
    data = data_utils.ld_dataset(transform)

    net = torchvision.models.resnet50()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)  # transfers to gpu if available

    optimizer_name = project_utils.query_optim()
    if optimizer_name == 'LA-SGD':
        optimizer = Lookahead_tutorial.Lookahead(torch.optim.SGD(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-Adam':
        optimizer = Lookahead_tutorial.Lookahead(torch.optim.Adam(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-ExtraSGD':
        optimizer = Lookahead_tutorial.Lookahead(extragradient.ExtraSGD(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-ExtraAdam':
        optimizer = Lookahead_tutorial.Lookahead(extragradient.ExtraAdam(net.parameters(), lr=1e-3))
    elif optimizer_name == 'LA-OGDA':
        optimizer = Lookahead_tutorial.Lookahead(OGDA.OGDA(net.parameters(), lr=1e-3))
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

