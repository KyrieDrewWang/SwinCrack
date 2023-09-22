""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear")
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, final=False):
        super().__init__()
        self.final = final
        self.upsample = upsample(2*in_channels, in_channels)
        if not final:
            self.conv0 = nn.Sequential(
                nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))
            self.conv1 = DoubleConv(in_channels, in_channels)
            self.conv2 = DoubleConv(in_channels, in_channels)
        else:
            self.conv0 = DoubleConv(2*in_channels, in_channels)
        
    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.concat((x, skip_x), dim=1)
        x = self.conv0(x)
        if self.final:
            return x
        x1 = self.conv1(x)
        x1 = x1 + x
        x2 = self.conv2(x1)
        x2 = x2 + x1
        return x2


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    

""" Full assembly of the parts to form the complete network """
class crackseg(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(crackseg, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.down1 = (Down(n_channels, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))
        self.down5 = (Down(512, 1024))       

        self.upsample5 = upsample(512, 1, 16)
        self.upsample4 = upsample(256, 1, 8)
        self.upsample3 = upsample(128, 1, 4)
        self.upsample2 = upsample(64, 1, 2)

        self.up5 = Up(512)
        self.up4 = Up(256)
        self.up3 = Up(128)
        self.up2 = Up(64)
        self.up1 = Up(32, final=True)

        self.skip0 = DoubleConv(3, 32)
        self.out = nn.Conv2d(32, 1, 3, 1, 1)
        
    def forward(self, x):
        x1 = self.skip0(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x5 = self.up5(x6, x5)
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)

        logits5 = self.upsample5(x5)
        logits4 = self.upsample4(x4)
        logits3 = self.upsample3(x3)
        logits2 = self.upsample2(x2)
        logits = self.out(x1)
        return logits, logits5, logits4, logits3, logits2, torch.zeros_like(logits)
    



if __name__ == "__main__":
    # model = crackseg()
    # input = torch.randn((1, 3, 480, 320))
    # output = model(input)
    # print([i.shape for i in output])
    
    inp = torch.randn(1, 3, 512, 512)
    model = crackseg()
    comp, param = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', comp))  # GFLOPs = GMac*2
    print('{:<30}  {:<8}'.format('Number of parameters: ', param))
    print("transformer have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000))
    
    # f, p = profile(model=model, inputs=(inp, ))
    out=model(inp)
    print(out[0].shape)