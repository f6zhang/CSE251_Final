import blur
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=="__main__":
    train_data, _ = blur.gaussian_blur_data()

    show_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=False)

    tbar = tqdm(enumerate(show_loader), total=len(show_loader), desc='Bar desc', leave=True)

    for iter, (inputs, labels) in tbar:
        id = 0
        for img in inputs:
            img = img.numpy()[0].astype(float)
            plt.imsave('./images/gauss_blur' + str(id) + '.jpg', img, cmap='gray')
            id += 1
        break

    train_data, _ = blur.move_blur_data()

    show_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=False)

    tbar = tqdm(enumerate(show_loader), total=len(show_loader), desc='Bar desc', leave=True)

    for iter, (inputs, labels) in tbar:
        id = 0
        for img in inputs:
            img = img.numpy()[0].astype(float)
            plt.imsave('./images/move_blur' + str(id) + '.jpg', img, cmap='gray')
            id += 1
        break