#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Reference:

DeepCrack: A deep hierarchical feature learning architecture for crack segmentation.
  https://www.sciencedirect.com/science/article/pii/S0925231219300566
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from thop import profile

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


class DeepCrackNet(nn.Module):
    def __init__(self, in_nc, num_classes, ngf, norm='batch'):
        super(DeepCrackNet, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv2 = nn.Sequential(*self._conv_block(ngf, ngf*2, norm_layer, num_block=2))
        self.side_conv2 = nn.Conv2d(ngf*2, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Sequential(*self._conv_block(ngf*2, ngf*4, norm_layer, num_block=3))
        self.side_conv3 = nn.Conv2d(ngf*4, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv4 = nn.Sequential(*self._conv_block(ngf*4, ngf*8, norm_layer, num_block=3))
        self.side_conv4 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv5 = nn.Sequential(*self._conv_block(ngf*8, ngf*8, norm_layer, num_block=3))
        self.side_conv5 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.fuse_conv = nn.Conv2d(num_classes*5, num_classes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        #self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        #self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=2, kernel_size=3, 
        stride=1, padding=1, bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv

    def forward(self, x):
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_conv1(conv1)
        side_output2 = self.side_conv2(conv2)
        side_output3 = self.side_conv3(conv3)
        side_output4 = self.side_conv4(conv4)
        side_output5 = self.side_conv5(conv5)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True) #self.up16(side_output5)

        fused = self.fuse_conv(torch.cat([side_output1, 
                                          side_output2, 
                                          side_output3,
                                          side_output4,
                                          side_output5], dim=1))
        return fused, side_output1, side_output2, side_output3, side_output4, side_output5

def define_deepcrack(in_nc, 
                     num_classes, 
                     ngf, 
                     norm='batch',
                     init_type='xavier', 
                     init_gain=0.02, 
                     ):
    net = DeepCrackNet(in_nc, num_classes, ngf, norm)
    return init_net(net, init_type, init_gain)


class deepseg(DeepCrackNet):
    
    def __init__(self,
                 in_nc=3, 
                 num_classes=1, 
                 ngf=64, 
                 norm='batch',
                 init_type='xavier', 
                 init_gain=0.02,) -> None:
        super(deepseg, self).__init__(in_nc, num_classes, ngf, norm)
        self.in_nc = in_nc
        self.num_classes = num_classes
        self.ngf = ngf
        self.norm = norm
        self.init_type = init_type
        self.init_gain = init_gain
        init_net(self, self.init_type, self.init_gain)
    
if __name__ == '__main__':
    a = torch.randn([1,3,224,224])
    # model = define_deepcrack(3, 1, 64)
    # b = model(a)
    model = deepseg()
    b = model(a)
    print(b[-1].shape)
    f, p = profile(model=model, inputs=(a, ))
    #out=model(inp)
    print(f, p)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))