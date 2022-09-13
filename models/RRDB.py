import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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


##########################################################################
##---------- Spatial Attention ----------
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3, bias=False):
        super(spatial_attn_layer, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        scale = self.spatial(x_compress)
        # scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=24, bias=True, act=nn.PReLU()):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                act,
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(self, nf, reduction=24, bias=False, act=nn.PReLU()):
        super(DAU, self).__init__()

        ## Spatial Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = ca_layer(nf,reduction, bias=bias, act=act)
        self.conv1x1 = nn.Conv2d(nf*2, nf, kernel_size=1, bias=bias)

    def forward(self, x):
        res = x
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nf, act=nn.ReLU()):
        super(DenseResidualBlock, self).__init__()

        def block(in_nf):
            layers = [nn.Conv2d(in_nf, nf, 3, 1, 1, bias=True)]
            layers += [nn.BatchNorm2d(nf)]
            layers += [act]
            return nn.Sequential(*layers)

        self.b1 = block(in_nf=1 * nf)
        self.b2 = block(in_nf=2 * nf)
        self.b3 = block(in_nf=3 * nf)
        self.b4 = block(in_nf=4 * nf)
        # self.b5 = block(in_nf=5 * nf)
        # self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.blocks = [self.b1, self.b2, self.b3, self.b4]

        self.dua_in = DAU(nf,act=act)
        self.dua_res = DAU(nf,act=act)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        out = self.dua_in(out) + self.dua_res(x)

        return out


class RRDB(nn.Module):
    """
    RRDB: Residual in Residual Dense Block
    """
    def __init__(self, nf, act=nn.ReLU()):
        super(RRDB, self).__init__()
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(nf,act=act), DenseResidualBlock(nf,act=act), DenseResidualBlock(nf,act=act)#, DenseResidualBlock(nf,act=act)
        )
        self.dua_in = DAU(nf,act=act)
        self.dua_res = DAU(nf,act=act)

    def forward(self, x):
        inputs = x
        out = self.dense_blocks(inputs)
        out = self.dua_in(out) + self.dua_res(x)

        return out


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
    def __init__(self, in_chan, out_chan, nf=64, act_fun='ReLU'):
        super(Net, self).__init__()

        """ ARCHITECTURE 
        in_chan: number of input channels
        out_chan: number of output channels
        nf: number of feature maps
        act_fun: activation function
        """

        if act_fun == 'PReLU':
            act = nn.PReLU()
        elif act_fun == 'SiLU':
            act = nn.SiLU()
        elif act_fun == 'mish':
            act = mish()
        else:
            act = nn.ReLU(inplace=False)

        self.features = nn.Sequential()

        self.features.add_module('inc', nn.Conv2d(in_chan, nf, kernel_size=3, stride=2, padding=1, bias=True))
        self.features.add_module('rbE1', RRDB(nf=nf,act=act))
        self.features.add_module('down1', down_samp(nf=nf,act=act))

        self.features.add_module('rbE2', RRDB(nf=nf,act=act))
        self.features.add_module('down2', down_samp(nf=nf,act=act))

        self.features.add_module('rbC', RRDB(nf=nf,act=act))

        self.features.add_module('up2', up_samp(nf=nf,act=act))
        self.features.add_module('rbD2', RRDB(nf=nf,act=act))

        self.features.add_module('up1', up_samp(nf=nf,act=act))
        self.features.add_module('rbD1', RRDB(nf=nf,act=act))

        self.features.add_module('up0', up_samp(nf=nf,act=act))
        self.features.add_module('outc', nn.Conv2d(nf, out_chan, kernel_size=1, bias=True))

        self.features.add_module('dua1', DAU(nf,act=act))
        self.features.add_module('dua2', DAU(nf,act=act))
        self.features.add_module('dua3', DAU(nf,act=act))
        self.features.add_module('dua4', DAU(nf,act=act))
        self.features.add_module('dua5', DAU(nf,act=act))

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
        x6 = torch.cat((x6,self.features.dua5(x5_ori)),1)
        x7 = self.features.up2(x6)
        x8 = self.features.rbD2(x7+self.features.dua4(x4_ori))
        x8 = torch.cat((x8,self.features.dua3(x3_ori)),1)
        x9 = self.features.up1(x8)
        x10 = self.features.rbD1(x9+self.features.dua2(x2_ori))
        x10 = torch.cat((x10,self.features.dua1(x1_ori)),1)

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
