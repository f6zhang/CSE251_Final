import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader,TensorDataset
from torch import optim
from torch.autograd import Variable
from blur import gaussian_blur_data,  move_blur_data, original_data
from torch.utils.data.sampler import SubsetRandomSampler
from models import *
import argparse
import torchvision

def train(train_loader, args):
    model.train()

    num_epochs = args.num_epochs
    patient = args.patient
    best_loss = 1<<30
    iter_loss = 0
    training_loss = 0
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        training_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            output = model(images)             
            loss = loss_func(output, labels)
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()   
            iter_loss += loss.item()
            training_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, iter_loss/100.0))
                iter_loss = 0.0
        print("Training Loss is {:.4f}.".format(training_loss / total_step))
             
        val_loss, val_acc = evaluate(valid_loader)
        if val_loss < best_loss:
            save_model(model, path='./latest_model.pt')
            best_loss = val_loss
            patient = 0
        else:
            patient += 1
            if patient >=5:
                break

def evaluate(data_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            output= model(images)
            loss = loss_func(output, labels)
            pred_y = torch.argmax(output, 1)
            correct += (pred_y == labels).sum().item()
            total += len(labels)
            total_loss += loss.item()
    accuracy = correct / total
    total_loss /= len(data_loader)
    print('Accuracy of the model on the evaluate images: %.4f' % accuracy)
    print('Loss of the model on the evaluate images: %.4f' % total_loss)
    
    model.train()
    return total_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--data_type', default='original')
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--patient', default=5)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=0.0001)
    args = parser.parse_args()

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torchvision.datasets.FashionMNIST(root='./data', download=True)
    train_data, test_data = None, None
    if args.data_type == 'move':
        train_data, test_data = move_blur_data()
    elif args.data_type == 'gaussian':
        train_data, test_data = gaussian_blur_data()
    else:
        train_data, test_data = original_data()

    indices = list(range(60000))
    np.random.shuffle(indices)
    split = 5000

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, sampler=valid_sampler)
    test_loader =  DataLoader(test_data, num_workers=1, batch_size=args.batch_size)


    model= CNN()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)   
    loss_func = nn.CrossEntropyLoss()


    train(train_loader, args)

