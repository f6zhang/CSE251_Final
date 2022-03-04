import numpy as np
import torchvision

if __name__ == "__main__":
    torchvision.datasets.FashionMNIST(root='./data', download=True,)