import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from thop import profile


class PFE_module(nn.Module): 
    def __init__(self,inp,dilation=[1,2,3,5],squeeze_radio=2,group_kernel_size=3,group_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, inp//8, kernel_size=3, dilation=dilation[0],padding=1)
        self.conv2 = nn.Conv2d(inp, inp//8, kernel_size=3, dilation=dilation[1],padding=2)
        self.conv3 = nn.Conv2d(inp, inp//8, kernel_size=3, dilation=dilation[2],padding=3)
        self.conv4 = nn.Conv2d(inp, inp//8, kernel_size=3, dilation=dilation[3],padding=5) 
        self.conv5 = nn.Conv2d(inp//8,inp//8, kernel_size=3, padding=1,bias=False) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigomid = nn.Sigmoid()
        self.conv = nn.Conv2d(inp//2, inp//2, kernel_size=3, padding=1,bias=False)
        self.squeeze = nn.Conv2d(256, 256 // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(256 // 2, 256, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC = nn.Conv2d(256 // squeeze_radio, 256, kernel_size=1, bias=False)

    def forward(self, x):
        x1= self.conv1(x)
        x21= self.conv2(x)
        x12 = x1 + x21 
        x2 = self.conv5(x12)
        x32= self.conv3(x)
        x23= x32 + x2 
        x3 = self.conv5(x23)
        x43= self.conv4(x)
        x34 = x3 + x43
        x4 = self.conv5(x34)
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x6_0 = self.pool(x5)
        x6 = self.conv(x6_0)
        x7 = self.sigomid(x6)
        x8 =  self.squeeze(x5)
        x9 = self.GWC(x8)
        x10 = self.PWC(x8) 
        x11 = x9 + x10 
        x = x11 * x7
        return x
    







