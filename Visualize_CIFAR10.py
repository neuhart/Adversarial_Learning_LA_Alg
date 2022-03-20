import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #convert PIL image into tensor and normalize

batch_size = 4 #number of samples per batch

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#download training set, store into ./data and apply transform

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True) #load in training set

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
#download test set, store into ./data and apply transform

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False) #load in test set

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #transpose axes (0,1,2) to (1,2,0)
    # (90 degrees turn and making sure the colour values are in the third axis)
    plt.show()
    #plt.imshow(np.transpose(npimg, (0, 2, 1)))  # transpose axes (0,1,2) to (1,2,0)
    #plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))