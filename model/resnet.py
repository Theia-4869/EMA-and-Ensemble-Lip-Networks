import torch
import torch.nn as nn

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundMeanNorm, BoundLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = NormDistConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, mean_normalize=True)
        self.conv2 = NormDistConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, mean_normalize=True)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = NormDistConv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, mean_normalize=True)

        self.relu = BoundReLU()

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        out1 = self.relu(*self.conv1(*paras))
        out1 = self.conv2(*out1)
        out2 = paras
        if self.shortcut:
            out2 = self.shortcut(*out2)
        out = out1 + out2 
        preact = out
        out = self.relu(*out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = NormDistConv(in_planes, planes, kernel_size=1, bias=False, mean_normalize=True)
        self.conv2 = NormDistConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, mean_normalize=True)
        self.conv3 = NormDistConv(planes, self.expansion * planes, kernel_size=1, bias=False, mean_normalize=True)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = NormDistConv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, mean_normalize=True)

        self.relu = BoundReLU()

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        out1 = self.relu(*self.conv1(*paras))
        out1 = self.relu(*self.conv2(*out1))
        out1 = self.conv3(*out1)
        out2 = paras
        if self.shortcut:
            out2 = self.shortcut(*out2)
        out = out1 + out2 
        preact = out
        out = self.relu(*out)
        if self.is_last:
            return out, preact
        else:
            return out


class resNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(resNet, self).__init__()
        self.in_planes = 64

        self.conv = NormDistConv(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False, mean_normalize=True)
        self.relu = BoundReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = BoundAdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, NormDistConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.conv1(*paras)
        paras = self.relu(*paras)
        for layer in self.layer1:
            paras = layer(*paras)
        for layer in self.layer2:
            paras = layer(*paras)
        for layer in self.layer3:
            paras = layer(*paras)
        for layer in self.layer4:
            paras = layer(*paras)
        paras = self.avgpool(*paras)
        x = torch.flatten(x, 1)
        return paras


def resnet18(**kwargs):
    return resNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return resNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return resNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return resNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class ResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, input_dim, name='resnet50', head='mlp', feat_dim=128, num_classes=10):
        super(ResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        head = []
        if head == 'linear':
            head.append(BoundLinear(dim_in, feat_dim, bias=False))
            head.append(BoundReLU())
            head.append(BoundLinear(feat_dim, num_classes))
        elif head == 'mlp':
            head.append(BoundLinear(dim_in, dim_in, bias=False))
            head.append(BoundReLU())
            head.append(BoundLinear(dim_in, feat_dim, bias=False))
            head.append(BoundReLU())
            head.append(BoundLinear(feat_dim, num_classes))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.head = nn.ModuleList(head)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.encoder(*paras)
        for layer in self.head:
            paras = layer(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class ResNetFeature(nn.Module):
    """backbone + projection head"""
    def __init__(self, input_dim, name='resnet50', head='mlp', feat_dim=128):
        super(ResNetFeature, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        head = []
        if head == 'linear':
            head.append(BoundLinear(dim_in, feat_dim, bias=False))
        elif head == 'mlp':
            head.append(BoundLinear(dim_in, dim_in, bias=False))
            head.append(BoundReLU(inplace=True))
            head.append(BoundLinear(dim_in, feat_dim, bias=False))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.head = nn.ModuleList(head)
        self.out_features = feat_dim

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        paras = self.encoder(*paras)
        for layer in self.head:
            paras = layer(*paras)
        return paras
