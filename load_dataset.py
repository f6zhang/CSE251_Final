import torchvision
from torchvision import transforms

def load_dataset(name):
    if name == 'fashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', \
            train = True, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
        
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', \
            train = False, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
    elif name == 'SVHN':
        train_dataset = torchvision.datasets.SVHN(root='./data', \
            train = True, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
        
        test_dataset = torchvision.datasets.SVHN(root='./data', \
            train = False, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
    elif name == 'EMNIST':
        train_dataset = torchvision.datasets.EMNIST(root='./data', \
            train = True, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
        
        test_dataset = torchvision.datasets.EMNIST(root='./data', \
            train = False, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
    elif name == 'USPS':
        train_dataset = torchvision.datasets.USPS(root='./data', \
            train = True, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
        
        test_dataset = torchvision.datasets.USPS(root='./data', \
            train = False, download=True, \
            transform=transforms.Compose([
            transforms.ToTensor()])
        )
    else:
        raise NotImplementedError("Dataset is not supported!")
    
    return train_dataset, test_dataset