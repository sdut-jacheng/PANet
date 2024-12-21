import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / (float(2 * sigma ** 2))) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5)  
    _2D_window = torch.mm(_1D_window.unsqueeze(1), _1D_window.unsqueeze(1).t()).float().unsqueeze(0).unsqueeze(0) 
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu_img1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu_img2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu_img1_sq = mu_img1.pow(2)
    mu_img2_sq = mu_img2.pow(2)

    mu1_time_mu2 = mu_img1 * mu_img2
    sigma_img1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu_img1_sq
    sigma_img2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu_img2_sq
    sigma_12_sq = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_time_mu2
    C1 = 0.0001
    C2 = 0.0009    

    ssim_map = ((2 * mu1_time_mu2 + C1) * (2 * sigma_12_sq + C2)) / (
            (mu_img1_sq + mu_img2_sq + C1) * sigma_img1_sq + sigma_img2_sq + C2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(self.window_size, channel=self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())

        window = window.type_as(img1)

        self.window = window
        self.channel = channel
        return -torch.log(_ssim(img1, img2, window, self.window_size, channel, self.size_average))


def ssim(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def loss_ssim(img1, img2, window_size=11, size_average=True):
    loss = 1-(ssim(img1, img2, window_size, size_average))
    return loss
