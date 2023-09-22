# Author : Zhang Chong


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import os
import time
import argparse
import sys
sys.path.append('.')
from config.config_SegNet import Config as cfg
from ptflops import get_model_complexity_info

bn_momentum = 0.1  # BN层的momentum
# # 在神经网络中，参数默认是进行随机初始化的。
# # 如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
# # 如果设置初始化，则每次初始化都是固定的。
# torch.cuda.manual_seed(1)  # 设置随机种子

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.in_channels = in_channels

        self.enco1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        id = []

        x = self.enco1(x)
        x, id1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)  # return_indices=True，返回输出最大值的序号，对于上采样操作会有帮助
        id.append(id1)
        x = self.enco2(x)
        x, id2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        x = self.enco3(x)
        x, id3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        x = self.enco4(x)
        x, id4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        x = self.enco5(x)
        x, id5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)

        return x, id


# 编码器+解码器
class SegNet(nn.Module):
    def __init__(self, 
                 in_channels=cfg.IN_CHANNELS, 
                 out_channels=cfg.OUT_CHANNELS):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = Encoder(self.in_channels)

        self.deco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deco4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deco1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.OutConv = nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x, id = self.encoder(x)

        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x = self.deco5(x)
        x = F.max_unpool2d(x, id[3], kernel_size=2, stride=2)
        x = self.deco4(x)
        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        x = F.max_unpool2d(x, id[1], kernel_size=2, stride=2)
        x = self.deco2(x)
        x = F.max_unpool2d(x, id[0], kernel_size=2, stride=2)
        x = self.deco1(x)

        x = self.OutConv(x)

        return x, torch.ones_like(x, requires_grad=False), torch.ones_like(x, requires_grad=False), torch.ones_like(x, requires_grad=False), torch.ones_like(x, requires_grad=False), torch.ones_like(x, requires_grad=False)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    input = torch.rand(1,3,512,512)
    net = SegNet()
    output = net(input)
    print(output[0].shape)
    model = SegNet()
    test = torch.rand(1, 3, 512, 512)
    device = torch.device("cuda")
    model.to(device)
    test = test.cuda()
    out = model(test)
    out = out
    for inx, i in enumerate(out):
        print(str(inx)+"th's shape is: ", i.shape)
    comp, param = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', comp))  # GFLOPs = GMac*2
    print('{:<30}  {:<8}'.format('Number of parameters: ', param))
    print("transformer have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000))