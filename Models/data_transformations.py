import torchvision.transforms as transforms


def resnet_transform():
    """https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (image transformation for resnet)"""
    return transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  # convert PIL image into tensor and transform to match ResNet requirements (see hyperlink))


def standard_transform():
    """Basic Transformation which just converts the data into torch tensors"""
    return transforms.Compose([transforms.ToTensor()])
