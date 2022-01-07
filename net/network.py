import torch
import math
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from resnet import *
from l2norm import L2Norm


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
        self.flt = nn.Flatten(2)
        self.feat_reduced = nn.Sequential(
            MixerBlock(512*256, 768),
            nn.Linear(768, 256)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(512*256,256 ),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

        self.reg_mlp = nn.Sequential(
            MixerBlock(512*256,256 ),
            nn.Linear(256,1)
        )

        self.off_mlp = nn.Sequential(
            MixerBlock(512*256,256 ),
            nn.Linear(256,2)
        )

        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99/0.01))
        nn.init.constant_(self.reg_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)

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

        cat = torch.cat([p3, p4, p5], dim=1) #will return array (n,c,h,w)
        n,c,h,w = cat.shape
        feat = self.flt(cat) #n,c,h*w
        feat = feat.transpose(1,2)  #n,h*w,c

        feat = self.feat_reduced(feat) #768 to 256

        x_cls = self.pos_mlp(feat).transpose(1,2) #n,h*w,1  # We are transposing to match the initial dimensions (n,c,h*w)
        x_reg = self.reg_mlp(feat).transpose(1,2) #n,h*w,1
        x_off = self.off_mlp(feat).transpose(1,2) #n,h*w,2

        return x_cls.view(n,1,h,w), x_reg.view(n,1,h,w), x_off.view(n,2,h,w)

    # def train(self, mode=True):
    #     # Override train so that the training mode is set as we want
    #     nn.Module.train(self, mode)
    #     if mode:
    #         # Set fixed blocks to be in eval mode
    #         self.conv1.eval()
    #         self.layer1.eval()
    #
    #         # bn is trainable in CONV2
    #         def set_bn_train(m):
    #             class_name = m.__class__.__name__
    #             if class_name.find('BatchNorm') != -1:
    #                 m.train()
    #             else:
    #                 m.eval()
    #         self.layer1.apply(set_bn_train)


# to do
#
# constuct a bn fixed CSP
#
#
#
#
#

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

class CSPNet_mod(nn.Module):
    # This is Batchnorm fixed version of CSP
    # under construction !!!!!!!!!!!!!!!!!!!
    def __init__(self):
        super(CSPNet_mod, self).__init__()

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

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.feat_act = nn.ReLU(inplace=True)

        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)

        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)

        nn.init.constant_(self.feat.bias, 0)
        nn.init.constant_(self.reg_conv.bias, -math.log(0.99/0.01))
        nn.init.constant_(self.pos_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)

        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)

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

        cat = torch.cat([p3, p4, p5], dim=1)

        feat = self.feat(cat)
        feat = self.feat_act(feat)

        x_cls = self.pos_conv(feat)
        x_cls = torch.sigmoid(x_cls)
        x_reg = self.reg_conv(feat)
        x_off = self.off_conv(feat)

        return x_cls, x_reg, x_off

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.layer2.apply(set_bn_eval)
            self.layer3.apply(set_bn_eval)
            self.layer4.apply(set_bn_eval)
