import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def imshow(dataloader, batch_size, classes, inv_transform=None):
    """Plot a batch of images
    dataloader: dataloader from which to get images
    classes: tuple/list of classes
    inv_transform (OPTIONAL): inversion of transformation
    """
    data_iter = iter(dataloader)
    images, labels = data_iter.next()
    img = torchvision.utils.make_grid(images)
    if inv_transform is not None:
        img = inv_transform(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose axes (0,1,2) to (1,2,0)
    # (90 degrees turn picture stack and making sure the colour values are in the third axis)
    plt.show()
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))  # print labels


def yes_no_check(question):
    """
    Ask a 'yes or no'-question
    return True for yes and False for no
    """
    x = input(question + ' (y/n):')
    while x not in ['y', 'n']:
        print('expected "y" or "n" ')
        x = input(question + ' (y/n):')
    return True if x == 'y' else False


def int_query(query):
    """
    Query an integer
    """
    x = input(query + ' (int):')
    while isinstance(x, int) is False:
        try:
            x = int(x)
        except:
            print('Integer expected')
            x = input(query + ' (int):')
    return x


def save_results(optimizer, bool_adv_train, results):
    filename = 'data/adv_results.csv' if bool_adv_train else 'data/clean_results.csv'
    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.Series(results, name=get_optim_name(optimizer))], axis=1)
    df.to_csv(filename, index=False)


def query_optim():
    implemented_optims = ['LA-SGD', 'LA-Adam', 'LA-ExtraAdam', 'LA-ExtraSGD', 'SGD', 'Adam', 'ExtraSGD', 'ExtraAdam']
    x = input('Select an opimizer {}:'.format(implemented_optims))
    assert x in implemented_optims, 'Implemented optimizer: {}'.format(implemented_optims)
    return x


def get_optim_name(optimizer):
    optimizer_name = optimizer.__class__.__name__
    if optimizer_name == 'Lookahead':
        optimizer_name += '-' + optimizer.optimizer.__class__.__name__
    return optimizer_name
