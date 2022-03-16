from turtle import forward
import torch
import torch.nn as nn
from torch import einsum
import numpy as np 
from einops import rearrange, repeat


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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
        self.down1 = nn.Conv2d(64, 64, kernel_size=9, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=9, stride=1, padding=4)
        self.down2 = nn.Conv2d(128, 128, kernel_size=9, stride=2, padding=4)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=9, stride=1, padding=4)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=9, stride=2, padding=4, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=9, stride=1, padding=4)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=9, stride=1, padding=4)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(self.bn1(self.relu(self.down1(self.relu(x)))))
        out = self.relu(self.up1(self.relu(self.conv3(self.bn2(self.relu(self.down2(self.relu(x1))))))))
        out = torch.cat([out, x1], dim=1)
        out = self.relu(self.up2(self.relu(self.conv4(self.bn3(out)))))
        out = torch.cat([out, x], dim=1)
        out = self.relu(self.conv6(self.relu(self.conv5(self.bn4(out)))))
        return out


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=2,
                 downscaling_factors=(2, 1, 1, 1), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(channels, num_classes, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(channels=channels, num_classes=num_classes, hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
