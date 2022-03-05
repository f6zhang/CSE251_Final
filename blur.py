import torchvision
from torchvision import transforms
from random import randrange, choice
import numpy as np
import torch
from scipy.ndimage import convolve


def get_dataset(name):
    if name == 'fashionMNIST':
        dataset = torchvision.datasets.FashionMNIST        
    elif name == 'SVHN':
        dataset = torchvision.datasets.SVHN
    elif name == 'EMNIST':
        dataset = torchvision.datasets.EMNIST
    elif name == 'USPS':
        dataset = torchvision.datasets.USPS
    else:
        raise NotImplementedError("Dataset is not supported!")
    return dataset
    

def original_data(dataset='fashionMNIST'):
    dataset_fn = get_dataset(name=dataset)
    train_data = dataset_fn(
        root='./data',
        train = True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_data = dataset_fn(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    return train_data, test_data

def gaussian_blur_data(dataset='fashionMNIST', kernal_size=5, sigma=2):
    dataset_fn = get_dataset(name=dataset)

    train_data = dataset_fn(
        root='./data',
        train = True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma)
        ])
    )

    test_data = dataset_fn(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma)
        ])
    )

    return train_data, test_data

def move_blur_data(dataset='fashionMNIST', kernal_size=5, sigma=2, box_size=10):
    dataset_fn = get_dataset(name=dataset)
    train_data = dataset_fn(
        root='./data',
        train = True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma),
            move_blur(box_size),
        ])
    )

    test_data = dataset_fn(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma),
            move_blur(box_size),
        ])
    )
    return train_data, test_data

class move_blur:
    def __init__(self, box_size):
        self.box_size = box_size

    def __call__(self, image):
        filter = np.zeros([self.box_size, self.box_size])
        left = choice([False, True])
        if left:
            for i in range(0, self.box_size):
                filter[i, -i-1] = 1 / self.box_size
        else:
            for i in range(0, self.box_size):
                filter[-i-1,i] = 1 / self.box_size
        image = convolve(image[0], filter, mode="constant", cval=0)
        return image[None,...]