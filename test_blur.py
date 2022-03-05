import blur
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

if __name__=="__main__":
    train_data, _ = blur.gaussian_blur_data()
    show_set, _ = torch.utils.data.random_split(train_data, [10, 59990])

    show_loader = DataLoader(dataset=show_set, batch_size=10, shuffle=False)

    tbar = tqdm(enumerate(show_loader), total=len(show_loader), desc='Bar desc', leave=True)

    for iter, (inputs, labels) in tbar:
        id = 0
        for img in inputs:
            img = img.numpy()[0]
            img = Image.fromarray(img, 'L')
            img.save('./images/gauss_blur' + str(id) + '.jpg')
            id += 1

    train_data, _ = blur.move_blur_data()
    show_set, _ = torch.utils.data.random_split(train_data, [10, 59990])

    show_loader = DataLoader(dataset=show_set, batch_size=10, shuffle=False)

    tbar = tqdm(enumerate(show_loader), total=len(show_loader), desc='Bar desc', leave=True)

    for iter, (inputs, labels) in tbar:
        id = 0
        for img in inputs:
            img = img.numpy()[0]
            img = Image.fromarray(img, 'L')
            img.save('./images/move_blur' + str(id) + '.jpg')
            id += 1