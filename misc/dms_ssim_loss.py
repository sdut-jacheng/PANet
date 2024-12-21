import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import pdb

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def cal_ms_ssim(img1, img2, level=5, window_size=5, dilations=[1,1,1,1,1]):
    img1 = (img1 + 1e-8) / (img1.max() + 1e-8)
    img2 = (img2 + 1e-8) / (img2.max() + 1e-8)
    img1 = img1 * 255.0
    img2 = img2 * 255.0
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    mssim = []
    cs = []
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        weights = weights.cuda(img1.get_device())
    window = window.type_as(img1)
    for i in range(level):
        kernel_size = window_size + (dilations[i] - 1) * (window_size - 1) - 1
        mu1 = F.conv2d(img1, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i])
        mu2 = F.conv2d(img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu2_sq
        sigma12 = F.conv2d(img1*img2, Variable(window, requires_grad=False), padding = kernel_size//2, dilation=dilations[i]) - mu1_mu2
        
        C1 = (0.01*255)**2
        C2 = (0.03*255)**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        ssim_value = ssim_map.mean()
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs_value = torch.mean(v1 / v2)
        
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
        mssim.append(ssim_value)
        cs.append(cs_value)
    mssim = torch.stack(mssim, dim=0)
    cs = torch.stack(cs, dim=0)
    weights = Variable(weights, requires_grad=False)
    ms_ssim = torch.prod(mssim**weights)
    return 1-ms_ssim
    
    
def cal_dms_ssim_loss(img1, img2, level=5, window_size=5, dilations=[1,2,3,6,9]):
    
    return cal_ms_ssim(img1, img2, level, window_size, dilations)