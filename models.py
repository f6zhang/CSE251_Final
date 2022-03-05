import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, inchannel, image_size, n_classes):
        super(CNN, self).__init__()
        self.image_size = image_size
        self.inchannel = inchannel
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=inchannel, out_channels=16, kernel_size=5, stride=1, padding=2),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        self.out = nn.Linear(32 * (image_size//4) * (image_size//4), n_classes)
        
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, self.inchannel, self.image_size, self.image_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

def save_model(model, path='./latest_model.pt'):
    model_dict = model.state_dict()
    state_dict = {'model': model_dict}
    torch.save(state_dict, path)
    
def load_model(model, device, path='./latest_model.pt'):
    model.load_state_dict(torch.load(path)["model"])
    model.to(device)
    return model