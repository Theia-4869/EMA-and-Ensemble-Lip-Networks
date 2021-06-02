import torch.nn as nn

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundMeanNorm, BoundLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d

class AlexNet(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(AlexNet, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 96, 7, 2, 2, bias=False))
        conv1.append(BoundReLU())
        conv1.append(BoundMaxPool2d(3, 2, 0))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(96, 256, 5, 1, 2, bias=False))
        conv2.append(BoundReLU())
        conv2.append(BoundMaxPool2d(3, 2, 0))
        self.conv2 = nn.ModuleList(conv2)

        conv3 = []
        conv3.append(NormDistConv(256, 384, 3, 1, 1, bias=False))
        conv3.append(BoundReLU())
        self.conv3 = nn.ModuleList(conv3)

        conv4 = []
        conv4.append(NormDistConv(384, 384, 3, 1, 1, bias=False))
        conv4.append(BoundReLU())
        self.conv4 = nn.ModuleList(conv4)

        conv5 = []
        conv5.append(NormDistConv(384, 256, 3, 1, 1, bias=False))
        conv5.append(BoundReLU())
        self.conv5 = nn.ModuleList(conv5)

        fc = []
        fc.append(BoundLinear(256 * 3 * 3, 1024, bias=False))
        fc.append(BoundReLU())
        fc.append(BoundLinear(1024, 512, bias=False))
        fc.append(BoundReLU())
        fc.append(BoundLinear(512, num_classes))
        self.fc = nn.ModuleList(fc)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv1:
            paras = layer(*paras)
        for layer in self.conv2:
            paras = layer(*paras)
        for layer in self.conv3:
            paras = layer(*paras)
        for layer in self.conv4:
            paras = layer(*paras)
        for layer in self.conv5:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class AlexNetFeature(nn.Module):
    def __init__(self, input_dim):
        super(AlexNetFeature, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 96, 7, 2, 2, bias=False))
        conv1.append(BoundReLU())
        conv1.append(BoundMaxPool2d(3, 2, 0))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(96, 256, 5, 1, 2, bias=False))
        conv2.append(BoundReLU())
        conv2.append(BoundMaxPool2d(3, 2, 0))
        self.conv2 = nn.ModuleList(conv2)

        conv3 = []
        conv3.append(NormDistConv(256, 384, 3, 1, 1, bias=False))
        conv3.append(BoundReLU())
        self.conv3 = nn.ModuleList(conv3)

        conv4 = []
        conv4.append(NormDistConv(384, 384, 3, 1, 1, bias=False))
        conv4.append(BoundReLU())
        self.conv4 = nn.ModuleList(conv4)

        conv5 = []
        conv5.append(NormDistConv(384, 256, 3, 1, 1, bias=False))
        conv5.append(BoundReLU())
        self.conv5 = nn.ModuleList(conv5)

        fc = []
        fc.append(BoundLinear(256 * 3 * 3, 1024, bias=False))
        fc.append(BoundReLU())
        fc.append(BoundLinear(1024, 512, bias=False))
        self.fc = nn.ModuleList(fc)
        self.out_features = 512

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv1:
            paras = layer(*paras)
        for layer in self.conv2:
            paras = layer(*paras)
        for layer in self.conv3:
            paras = layer(*paras)
        for layer in self.conv4:
            paras = layer(*paras)
        for layer in self.conv5:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        return paras
