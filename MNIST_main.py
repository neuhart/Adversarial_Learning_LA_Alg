import torch
import project_utils
import data_utils
from absl import app
from Models import models, data_transformations
"""
1) https://pytorch.org/hub/pytorch_vision_resnet/
data = data_utils.ld_dataset(transform=data_transformations.resnet_transform())
net = torchvision.models.resnet50()  # 1)
"""


def main(_):
    data_utils.code_settings()  # specify general settings
    device = torch.device(project_utils.int_query('Select GPU [0,3]:')) if torch.cuda.is_available() else torch.device('cpu')

    data = data_utils.ld_dataset(dataset_name='MNIST', transform=data_transformations.standard_transform())

    # query which optimizers to use for training
    optims_list = project_utils.get_optims()

    for optim in optims_list:

        net = models.MNIST_CNN()
        net.to(device)  # transfers to gpu if available

        # Determine which optimizer to use
        optimizer = project_utils.set_optim(optim=optim, model=net)

        # Train model
        net.train()
        data_utils.my_training(data.train, net, optimizer, device)

        # Evaluation
        net.eval()
        data_utils.my_evaluation(data.test, net, optimizer, device)


if __name__ == "__main__":
    app.run(main)

