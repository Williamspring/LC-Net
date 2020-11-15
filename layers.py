import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class MultiInput(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, inter_channels,pool_size):
        super(MultiInput, self).__init__()
        #金字塔池化
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])


        # inter_channels = int(in_channels/4)
        self.conv1_1_0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv1_1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv1_1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))


        self.conv = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1_1 = self.conv1_1_0(x)
        x1_2 = F.interpolate(self.conv1_1_1(self.pool1(x)), (h, w),mode="bilinear")
        x1_3 = F.interpolate(self.conv1_1_2(self.pool2(x)), (h, w),mode="bilinear")

        x1 = self.conv(x1_1 + x1_2 + x1_3)

        return x1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class LCM(nn.Module):
    def __init__(self,in_ch,ratio=2,gama =0.2):

        super(LCM,self).__init__()

        self.conv0 = nn.Conv2d(in_ch,in_ch,1,1)
        self.conv1 = nn.Conv2d(in_ch,1,1)
        self.conv2 = nn.Conv2d(in_ch,in_ch,1,1)


        self.muti_conv = nn.Sequential(
            nn.Conv2d(in_ch,in_ch//ratio,1),
            nn.BatchNorm2d(in_ch//ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//ratio,in_ch,1),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gama

    def forward(self, x,label):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(label, (h, w))
        y = y.expand_as(x)
        g = F.adaptive_avg_pool2d(y, (1, 1))

        input_conv = self.conv0(x)
        # input_ = x / x.max()
        input_ = input_conv / input_conv.max()
        gc = torch.pow((input_ - g),2)

        k = gc.min()
        wg = torch.exp(-(gc - k) / self.gamma)

        context_mask = wg*input_conv

        #squeeze
        gc_input = F.adaptive_avg_pool2d(x, (1, 1))+ F.adaptive_avg_pool2d(context_mask, (1, 1))

        se = self.muti_conv(gc_input)

        out = x*se+context_mask
        
        return out
