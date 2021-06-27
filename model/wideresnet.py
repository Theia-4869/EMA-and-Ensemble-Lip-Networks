import torch
import torch.nn as nn
import torch.nn.functional as F

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundTanh, BoundMeanNorm, BoundLinear, BoundFinalLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d, BoundDropout

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = NormDistConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, mean_normalize=True)
        self.conv2 = NormDistConv(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, mean_normalize=True)
        self.drop_rate = drop_rate
        if drop_rate > 0:
            self.dropout = BoundDropout(drop_rate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and NormDistConv(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False, mean_normalize=True) or None

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        out1 = self.conv1(*paras)
        if self.drop_rate > 0:
            out1 = self.dropout(*out1)
        out1 = self.conv2(*out1)
        out2 = paras
        if self.convShortcut is not None:
            out2 = self.convShortcut(*out2)
        if out1[1] is None or out1[2] is None or out2[1] is None or out2[2] is None:
            out = (out1[0] * 0.8 + out2[0] * 0.2, None, None)
        else:
            out = (out1[0] * 0.8 + out2[0] * 0.2, out1[1] * 0.8 + out2[1] * 0.2, out1[2] * 0.8 + out2[2] * 0.2)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate))
        return nn.ModuleList(layers)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.layer:
            paras = layer(*paras)
        return paras


class wideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0):
        super(wideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # conv before any network block
        self.conv = NormDistConv(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False, mean_normalize=True)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        # avgpool before fc layer
        self.avgpool = BoundAdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, NormDistConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, NormDist):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.conv(*paras)
        paras = self.block1(*paras)
        paras = self.block2(*paras)
        paras = self.block3(*paras)
        paras = self.avgpool(*paras)
        return paras


class WideResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, input_dim, feat_dim=512, num_classes=10):
        super(WideResNet, self).__init__()
        self.encoder = wideResNet()
        dim_in = 128
        head = []
        head.append(NormDist(dim_in, dim_in, bias=False, mean_normalize=True))
        head.append(NormDist(dim_in, feat_dim, bias=False, mean_normalize=True))
        head.append(NormDist(feat_dim, num_classes, bias=True, mean_normalize=False))
        self.head = nn.ModuleList(head)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.encoder(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.head:
            paras = layer(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class WideResNetFeature(nn.Module):
    """backbone + projection head"""
    def __init__(self, input_dim, feat_dim=512):
        super(WideResNetFeature, self).__init__()
        self.encoder = wideResNet()
        dim_in = 128
        head = []
        head.append(NormDist(dim_in, dim_in, bias=False, mean_normalize=True))
        head.append(NormDist(dim_in, feat_dim, bias=False, mean_normalize=True))
        self.head = nn.ModuleList(head)
        self.out_features = feat_dim

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.encoder(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.head:
            paras = layer(*paras)
        return paras
