# Author : Zhang Chong

import os   
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from config.config_UNet import Config as cfg
from ptflops import get_model_complexity_info

""" Parts of the U-Net model """

# 卷积层
# 连续2次卷积操作
# 卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。
# [BN]: Batch Normalization 批归一化
# ReLU: 激活函数
class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU)*2'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()  # 调用父类的构造函数
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    '''
    这里的x是Tensor(Batch_Size x Channels x Height x Width),表示图像（需要做卷积操作的图像） 
    double_conv(x)返回结果也是Tensor(Batch_Size2 x Channels2 x Height2 x Width2)
    '''
    def forward(self, x):
        return self.double_conv(x)


# 池化层 压缩数据和参数的量 提高计算速度和鲁棒性
class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2表示池化窗口的大小
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 上采样模块，包括上采样操作和行特征的融合（就是将feature map的通道进行叠加，俗称Concat）
# Concat融合的两个feature map（Unet左边和右边对应的特征图）的大小不一定相同，UNet采用的Concat方案是将小的feature map进行padding
# padding的方式是补0，一种常规的常量填充。
class Up(nn.Module):
    """Upscaling then double conv"""

    # 两种上采样的方法：Upsample和ConvTranspose2d，也就是双线性插值和反卷积。
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:  # 双线性插值
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    '''
    前向传播函数
    x1 接收的是上采样的数据
    x2 接收的是特征融合的数据
    特征融合方法就是，上文提到的，先对小的feature map进行padding，再进行concat。
    '''
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is N x C x H x W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)   # 在给定维度(dim=1)上对输入的张量序列seq进行连接操作
        return self.conv(x)


# 最后输出那次用到的卷积操作
# UNet网络的输出需要根据分割数量，整合输出通道
# 操作很简单，就是channel的变换
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




""" Full assembly of the parts to form the complete network """

# 根据UNet网络的结构，设置每个模块的输入输出通道个数以及调用顺序
class UNet(nn.Module):
    # n_channels是样本影像的通道数；n_classes是你要分为的类数目
    def __init__(self, 
                 n_channels=cfg.N_CHANNELS, 
                 n_classes=cfg.N_CLASSES, 
                 bilinear=cfg.BILINEAR):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.drop3 = nn.Dropout2d(0.5)
        self.down4 = Down(512, 1024)
        # self.drop4 = nn.Dropout2d(0.5)
        self.up1 = Up(1024, 512, self.bilinear)
        self.up2 = Up(512, 256, self.bilinear)
        self.up3 = Up(256, 128, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    # x是样本真值影像/输入
    def forward(self, x):
        x_inc = self.inc(x)
        x1 = self.down1(x_inc)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x3 = self.drop3(x3)
        x4 = self.down4(x3)
        # x4 = self.drop3(x4)
        x_up = self.up1(x4, x3)
        x_up = self.up2(x_up, x2)
        x_up = self.up3(x_up, x1)
        x_up = self.up4(x_up, x_inc)

        x_out = self.outc(x_up)  # 二分类

        return x_out, torch.ones_like(x_out), torch.ones_like(x_out), torch.ones_like(x_out), torch.ones_like(x_out), torch.ones_like(x_out)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    net = UNet()
    input = torch.rand(1,3,512,512)
    output = net(input)
    print(net)
    print(output[0].shape)
    model = UNet()
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