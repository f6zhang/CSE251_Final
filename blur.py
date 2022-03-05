import torchvision
from torchvision import transforms
from random import randint
import numpy as np

def gaussian_blur_data(kernal_size=5, sigma=2):
    train_data = torchvision.datasets.FashionMNIST(
        root='./data',
        train = True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma)
        ])
    )

    test_data = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma)
        ])
    )

    return train_data, test_data

def move_blur_data(kernal_size=5, sigma=2, box_size=5):
    train_data = torchvision.datasets.FashionMNIST(
        root='./data',
        train = True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma),
            move_blur(box_size)
        ])
    )

    test_data = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.GaussianBlur(kernal_size, sigma),
            move_blur(box_size)
        ])
    )
    return train_data, test_data

class move_blur:
    def __init__(self, box_size):
        self.box_size = box_size

    def __call__(self, image):
        random_x = randint(-self.box_size, self.box_size)
        random_y = randint(-self.box_size, self.box_size)
        for i in range(np.max((0, random_x)), np.min((28, image.shape[1] + random_x))):
            for j in range(np.max((0, random_y)), np.min((28, image.shape[2] + random_y))):
                image[0, i, j] = (image[0, i, j] + image[0, i-random_x, j-random_y])/2
        return image