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
from scipy.ndimage import gaussian_filter
from random import randint,choice
import random


np.random.seed(2048)
torch.manual_seed(2048)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases
        
def move_blur(image, box_size=5):
  random_x, random_y = 0, 0
  while random_x + random_y < box_size:
    random_x = randint(1, box_size)
    random_y = randint(1, box_size)
  filter = np.zeros((random_x, random_y))
  steps = max((random_x, random_y))
  left = choice([True, False])
  for i in range(steps):
      if left:
          filter[int(random_x/steps * i), int(random_y/steps * i)] = 1/steps
      else:
          filter[random_x - int(random_x / steps * i) - 1, int(random_y / steps * i)] = 1 / steps
  return convolve(image, np.expand_dims(filter, axis=0), mode='constant', cval=0.0)
  
  
def train(train_loader, args):
    start_time = time.time()
    model.train()

    num_epochs = args.num_epochs
    patient = args.patient
    best_loss = 1<<30
    iter_loss = 0
    training_loss = 0
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        training_loss = 0
        for i, (images, _) in enumerate(train_loader):
            inputs = torch.tensor(move_blur(images.squeeze(axis=1).detach().numpy(), box_size=10), device=device).unsqueeze(axis=1)
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
        print("Training Loss is {:.6f}.".format(training_loss / total_step))
             
        val_loss = evaluate(valid_loader)
        if val_loss < best_loss:
            torch.save(model, './' + "restore_move_"+ args.data +  "_" +'latest_model.pt')
            best_loss = val_loss
            patient = 0
        else:
            patient += 1
            if patient >=5:
                break

def evaluate(data_loader):
    model.eval()
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, _ in data_loader:
            inputs = torch.tensor(move_blur(images.squeeze(axis=1).detach().numpy(), box_size=10), device=device).unsqueeze(axis=1)
            targets = images.to(device)
            output = model(inputs)             
            loss = loss_func(output, targets)
            total += len(targets )
            total_loss += loss.item()
    total_loss /= len(data_loader) 
    print('Loss of the model on the evaluate images: %.6f' % total_loss)
    
    model.train()
    return total_loss

def test(model, data_loader):
    model = torch.load('./restore_move_' + args.data +  "_" +'latest_model.pt')
    model.eval()
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, _ in data_loader:
            inputs = torch.tensor(move_blur(images.squeeze(axis=1).detach().numpy(), box_size=10), device=device).unsqueeze(axis=1)
            targets = images.to(device)
            output = model(inputs)             
            loss = loss_func(output, targets)
            total += len(targets)
            total_loss += loss.item()
                
    total_loss /= len(data_loader) 
    print('Loss of the model on the Test images: %.6f' % total_loss)

    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--data', type=str, default='FashionMNIST')
    parser.add_argument('--data_type', type=str, default='original')
    parser.add_argument('--num_epochs', type=int,default=10)
    parser.add_argument('--patient', type=int,  default=5)
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--isRestore', type=bool, default=False)
    args = parser.parse_args()
    
    print("Use GPU: " + str(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = original_data(dataset=args.data)

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
    model.apply(init_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)   
    loss_func = nn.MSELoss()

    print("Old!")
    train(train_loader, args)
    test(model, test_loader)