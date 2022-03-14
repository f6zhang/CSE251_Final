import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader,TensorDataset
from torch import optim
from torch.autograd import Variable
from blur import *
from torch.utils.data.sampler import SubsetRandomSampler
from models import *
import argparse
import torchvision
import time
import matplotlib.pyplot as plt

np.random.seed(2048)
torch.manual_seed(2048)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

def train(train_loader, args):
    start_time = time.time()
    model.train()

    num_epochs = args.num_epochs
    patient = args.patient
    best_loss = 1<<30
    train_losses = []
    val_losses = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        training_loss = 0
        iter_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = random_rotate(images)
            inputs = torch.tensor(blur_filter(images), device=device, dtype=torch.float32)
            targets = images.to(device)
            output = model(inputs)             
            loss = loss_func(output, targets)
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()   
            iter_loss += loss.item()
            training_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time cost: {:.2f}s' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, iter_loss/100.0, time.time()-start_time))
                iter_loss = 0.0
        print("Training Loss is {:.4f}.".format(training_loss / total_step))
        train_losses.append(training_loss / total_step)

        val_loss = evaluate(valid_loader)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            torch.save(model, './' + "restore_"+ args.data + "_" + str(args.data_type) +  "_" +'latest_model.pt')
            best_loss = val_loss
            patient = 0
        else:
            patient += 1
            if patient >=5:
                break
    show_plot(train_losses, val_losses, type="Loss", save_path="results/move/loss_plt.png")

def evaluate(data_loader):
    model.eval()
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, _ in data_loader:
            inputs = torch.tensor(blur_filter(images), device=device, dtype=torch.float32)
            targets = images.to(device)
            output = model(inputs)             
            loss = loss_func(output, targets)
            total += len(targets )
            total_loss += loss.item()
    total_loss /= len(data_loader)
    print('Loss of the model on the evaluate images: %.4f' % total_loss)
    
    model.train()
    return total_loss

def test(model, data_loader):
    model = torch.load('./' + "restore_"+ args.data + "_" + str(args.data_type) +  "_"+'latest_model.pt')
    model.eval()
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, _ in data_loader:
            inputs = torch.tensor(blur_filter(images), device=device, dtype=torch.float32)
            targets = images.to(device)
            output = model(inputs)             
            loss = loss_func(output, targets)
            total += len(targets)
            total_loss += loss.item()
    total_loss /= len(data_loader)
    print('Loss of the model on the Test images: %.4f' % total_loss)

    return total_loss


# Draw the plot for train/validation loss/accuarcy
def show_plot(train, validation, save_path=None, type="Loss"):
    x_axis = []
    for i in range(len(train)):
        x_axis.append(i + 1)

    plt.figure()
    plt.plot(x_axis, train, label="Train")
    plt.plot(x_axis, validation, label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.title("Train and Validation " + type + " over epoches")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--data', type=str, default='EMNIST')
    parser.add_argument('--data_type', type=str, default='original')
    parser.add_argument('--num_epochs', type=int,default=10)
    parser.add_argument('--patient', type=int,  default=5)
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--box_size', type=int, default=9)
    parser.add_argument('--kernel', type=int, default=9)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--isRestore', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    print("Use GPU: " + str(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = original_data(dataset=args.data)

    if args.data_type == 'move':
        blur_filter = move_blur(args.box_size, args.kernel, args.sigma)
    elif args.data_type == 'gaussian':
        blur_filter = torchvision.transforms.GaussianBlur(args.kernel, args.sigma)
    else:
        blur_filter = None

    random_rotate = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=180)
                ])

    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    split = len(train_data) // 10

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, sampler=valid_sampler)
    test_loader =  DataLoader(test_data, num_workers=1, batch_size=args.batch_size)

    inchannel, n, n_classes = 1, 0, 0
    if len(train_data.data[0].shape) == 2:
        n, _ = train_data.data[0].shape
    elif len(train_data.data[0].shape) == 3:
        inchannel, n , _= train_data.data[0].shape
    try:
        n_classes = max(train_data.targets) + 1
    except:
        n_classes = max(train_data.labels) + 1

    
    model = RestoreCNN(inchannel, n)
    if args.load:
        model = torch.load('./' + "restore_" + 'EMNIST' + "_" + str(args.data_type) + "_" + 'latest_model.pt')
    else:
        model.apply(init_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)   
    loss_func = nn.MSELoss()

    train(train_loader, args)
    test(model, test_loader)