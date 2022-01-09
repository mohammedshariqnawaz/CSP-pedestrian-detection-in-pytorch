import torch
import math
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from net.resnet import *
from .l2norm import L2Norm


class CSPNet(nn.Module):
    def __init__(self):
        super(CSPNet, self).__init__()

        resnet = resnet50(pretrained=True, receptive_keep=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        # Detection head
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(16, 768),
            nn.Linear(768, 256)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(16,256 ),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

        self.reg_mlp = nn.Sequential(
            MixerBlock(16,256 ),
            nn.Linear(256,1)
        )

        self.off_mlp = nn.Sequential(
            MixerBlock(16,256 ),
            nn.Linear(256,2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)

        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)

        x = self.layer4(x)
        p5 = self.p5(x)
        p5 = self.p5_l2(p5)

        cat = torch.cat([p3, p4, p5], dim=1) # (b,c,h,w)
        
        cat_permuted = cat.permute(0,2,3,1) #b,h,w,c
        b,h,w,c =cat_permuted.shape

        windows = window_partition(cat_permuted,4) # num_windows*B, window_size, window_size, C
        # windows = windows.permute(0,1,3) #h*w* b, p_s*p_s, c
        
        feat = self.mlp_with_feat_reduced(windows) #768 to 256

        x_cls = self.pos_mlp(feat).transpose(1,2) #b,h*w,1  # We are transposing to match the initial dimensions (n,c,h*w)
        x_reg = self.reg_mlp(feat).transpose(1,2) #b,h*w,1
        x_off = self.off_mlp(feat).transpose(1,2) #b,h*w,2

        x_cls = window_reverse(x_cls,4,h,w)
        x_reg = window_reverse(x_reg,4,h,w)
        x_off = window_reverse(x_off,4,h,w)

        return x_cls.permute(0,2,3,1), x_reg.permute(0,2,3,1), x_off.permute(0,2,3,1)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size* window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    print(windows.shape)
    print("B",B)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B,-1, H, W)
    return x
    
class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(num_channels)
        self.token_mix = MlpBlock(num_tokens, num_tokens*2)
        self.ln_channel = nn.LayerNorm(num_channels)
        self.channel_mix = MlpBlock(num_channels, num_channels*2)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x

