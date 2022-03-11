from turtle import forward
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


# Conv2d operation block, 
class conv_block(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch, image_size, n_classes):
        super(U_Net,self).__init__()
        
        self.UnetMaxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(inChannel=img_ch,outChannel=64)
        self.Conv2 = conv_block(inChannel=64,outChannel=128)
        self.Conv3 = conv_block(inChannel=128,outChannel=256)
        #self.Conv4 = conv_block(inChannel=256,outChannel=512)
        #self.Conv5 = conv_block(inChannel=512,outChannel=1024)

        #self.Up5 = up_conv(inChannel=1024,outChannel=512)
        #self.Up_conv5 = conv_block(inChannel=1024, outChannel=512)

        #self.Up4 = up_conv(inChannel=512,outChannel=256)
        #self.Up_conv4 = conv_block(inChannel=512, outChannel=256)
        
        self.Up3 = up_conv(inChannel=256,outChannel=128)
        self.Up_conv3 = conv_block(inChannel=256, outChannel=128)
        
        self.Up2 = up_conv(inChannel=128,outChannel=64)
        self.Up_conv2 = conv_block(inChannel=128, outChannel=64)

        #self.classifier = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
        self.out = nn.Linear(64 * (image_size) * (image_size), n_classes)



    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.UnetMaxpool(x1)
        x2 = self.Conv2(x2)      
        x3 = self.UnetMaxpool(x2)
        x3 = self.Conv3(x3)
        #x4 = self.UnetMaxpool(x3)
        #x4 = self.Conv4(x4)
        #x5 = self.UnetMaxpool(x4)
        #x5 = self.Conv5(x5)

        #x6 = self.Up5(x5)
        #x6 = torch.cat((x4,x6),dim=1)    
        #x6 = self.Up_conv5(x6)  
        #x7 = self.Up4(x6)
        #x7 = torch.cat((x3,x7),dim=1)
        #x7 = self.Up_conv4(x7)
        x8 = self.Up3(x3) #self.Up3(x7)
        x8 = torch.cat((x2,x8),dim=1)
        x8 = self.Up_conv3(x8)
        x9 = self.Up2(x8)
        x9 = torch.cat((x1,x9),dim=1)
        x9 = self.Up_conv2(x9)
        
        x9 = x9.view(x9.size(0), -1)       
        output  = self.out(x9)
        return output 

class RestoreCNN(nn.Module):
    def __init__(self, inchannel, image_size):
        super(RestoreCNN, self).__init__()
        self.image_size = image_size
        self.inchannel = inchannel
        
        self.up1 = up_conv(inChannel=inchannel,outChannel=32)
        self.up2 = up_conv(inChannel=32,outChannel=64)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down1 = conv_block(inChannel=64, outChannel=inchannel)
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.maxpool(x)
        x = self.down1(x)
        output = self.maxpool(x)

        return output

