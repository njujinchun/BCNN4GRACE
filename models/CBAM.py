""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
import torch
import torch.nn as nn


class mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=32, act=nn.ReLU()):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            act,
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channel, act=nn.ReLU()):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel,act=act)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, nf=64, bn=True, act=nn.ReLU()):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(1):
            modules_body.append(nn.Conv2d(nf, nf, kernel_size=3,stride=1,padding=1, bias=True))
            if bn: modules_body.append(nn.BatchNorm2d(nf))

        modules_body.append(CBAM(nf,act=act))
        modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class down_samp(nn.Module):
    def __init__(self, nf=64, act=nn.ReLU()):
        super(down_samp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, 1*nf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(1*nf),
            act,
        )

    def forward(self, x):
        return self.conv(x)


class up_samp(nn.Module):
    def __init__(self, nf=64, act=nn.ReLU()):
        super(up_samp, self).__init__()
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(2*nf, nf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf),
            act,
        )

    def forward(self, x):
        return self.convT(x)


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, act_fun='ReLU'):
        super(Net, self).__init__()

        if act_fun == 'PReLU':
            act = nn.PReLU()
        elif act_fun == 'SiLU':
            act = nn.SiLU()
        elif act_fun == 'mish':
            act = mish()
        else:
            act = nn.ReLU(inplace=False)

        self.features = nn.Sequential()
        self.features.add_module('inc', nn.Conv2d(in_channels, 1*nf,
                                                      kernel_size=3, stride=2, padding=1, bias=True))

        self.features.add_module('rbE1', RCAB(nf=nf,act=act))
        self.features.add_module('down1', down_samp(nf=nf,act=act))

        self.features.add_module('rbE2', RCAB(nf=nf,act=act))
        self.features.add_module('down2', down_samp(nf=nf,act=act))

        self.features.add_module('rbC', RCAB(nf=nf,act=act))

        self.features.add_module('up2', up_samp(nf=nf,act=act))
        self.features.add_module('rbD2', RCAB(nf=nf,act=act))

        self.features.add_module('up1', up_samp(nf=nf,act=act))
        self.features.add_module('rbD1', RCAB(nf=nf,act=act))

        self.features.add_module('up0', up_samp(nf=nf,act=act))
        self.features.add_module('outc', nn.Conv2d(nf, out_channels, kernel_size=1, bias=True))

    def forward(self, x):
        x1 = self.features.inc(x)
        x1_ori = x1

        x2 = self.features.rbE1(x1)
        x2_ori = x2
        x3 = self.features.down1(x2)
        x3_ori = x3
        x4 = self.features.rbE2(x3)
        x4_ori = x4
        x5 = self.features.down2(x4)
        x5_ori = x5

        x6 = self.features.rbC(x5)

        x6 = torch.cat((x6,x5_ori),1)
        x7 = self.features.up2(x6)
        x8 = self.features.rbD2(x7+x4_ori)
        x8 = torch.cat((x8,x3_ori),1)
        x9 = self.features.up1(x8)
        x10 = self.features.rbD1(x9+x2_ori)
        x10 = torch.cat((x10,x1_ori),1)

        x11 = self.features.up0(x10)
        y = self.features.outc(x11)

        return y

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
