import torchvision
from torchvision import transforms
from random import randint, choice
import numpy as np
from scipy.ndimage.filters import convolve
from copy import deepcopy

def get_dataset(name):
    if name == 'FashionMNIST':
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
    

def original_data(dataset='FashionMNIST'):
    dataset_fn = get_dataset(name=dataset)
    if dataset in ['FashionMNIST','USPS']:
        train_data = dataset_fn(
            root='./data',
            train = True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        test_data = dataset_fn(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        return train_data, test_data
    elif dataset in ['SVHN']:
        train_data = dataset_fn(
            root='./data',
            split = 'train',
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split = 'test',
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
        )

        return train_data, test_data
    elif dataset in ['EMNIST']:
        train_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train = True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        return train_data, test_data


def gaussian_blur_data(dataset='FashionMNIST', kernal_size=9, sigma=2):
    dataset_fn = get_dataset(name=dataset)

    if dataset in ['FashionMNIST','USPS']:
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

    elif dataset in ['SVHN']:
        train_data = dataset_fn(
            root='./data',
            split = 'train',
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                torchvision.transforms.GaussianBlur(kernal_size, sigma)
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split = 'test',
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                torchvision.transforms.GaussianBlur(kernal_size, sigma)
            ])
        )

        return train_data, test_data
    
    elif dataset in ['EMNIST']:
        train_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train = True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                 torchvision.transforms.GaussianBlur(kernal_size, sigma)
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                 torchvision.transforms.GaussianBlur(kernal_size, sigma)
            ])
        )
        return train_data, test_data

def move_blur_data(dataset='FashionMNIST', kernal_size=9, sigma=2, box_size=9):
    dataset_fn = get_dataset(name=dataset)
    if dataset in ['FashionMNIST','USPS']:
        train_data = dataset_fn(
            root='./data',
            train = True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )

        test_data = dataset_fn(
            root='./data',
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )
        return train_data, test_data
    elif dataset in ['SVHN']:
        train_data = dataset_fn(
            root='./data',
            split='train',
            download=False,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split='test',
            download=False,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )
        return train_data, test_data
    elif dataset in ['EMNIST']:
        train_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train = True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )

        test_data = dataset_fn(
            root='./data',
            split = 'balanced',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #torchvision.transforms.GaussianBlur(kernal_size, sigma),
                move_blur(box_size, kernal_size, sigma)
            ])
        )
        return train_data, test_data

class move_blur:
    def __init__(self, box_size=9, kernal_size=9, sigma=2):
        self.box_size = box_size
        self.gauss = torchvision.transforms.GaussianBlur(kernal_size, sigma)

    def __call__(self, ori_images):
        images = deepcopy(ori_images)
        output = np.zeros(images.shape)
        images = self.gauss(deepcopy(images))
        for i in range(images.shape[0]):
            random_x, random_y = 0, 0
            while random_x + random_y < self.box_size:
                random_x = randint(1, self.box_size)
                random_y = randint(1, self.box_size)
            filter = np.zeros((random_x, random_y))
            steps = max((random_x, random_y))
            left = choice([True, False])
            for j in range(steps):
                if left:
                    filter[int(random_x/steps * j), int(random_y/steps * j)] = 1/steps
                else:
                    filter[random_x - int(random_x / steps * j) - 1, int(random_y / steps * j)] = 1 / steps
            if len(images.shape) == 3:
                output[i] = convolve(images[i], filter, mode='constant', cval=0.0)
            elif len(images.shape) == 4:
                output[i] = convolve(images[i], np.expand_dims(filter, 0), mode='constant', cval=0.0)
            else:
                print("Error Move Blur")
        return output