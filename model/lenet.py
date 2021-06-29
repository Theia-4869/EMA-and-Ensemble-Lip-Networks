import torch.nn as nn

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundTanh, BoundMeanNorm, BoundLinear, BoundFinalLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d

class LeNet(nn.Module):
    def __init__(self, input_dim, hidden=512, num_classes=10):
        super(LeNet, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 6, 5, bias=False, mean_normalize=True))
        conv1.append(BoundMaxPool2d(2))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(6, 16, 5, bias=False, mean_normalize=True))
        conv2.append(BoundMaxPool2d(2))
        self.conv2 = nn.ModuleList(conv2)

        fc = []
        fc.append(NormDist(16 * 5 * 5, 120, bias=False, mean_normalize=True))
        fc.append(NormDist(120, hidden, bias=False, mean_normalize=True))
        fc.append(NormDist(hidden, num_classes, bias=True, mean_normalize=False))
        self.fc = nn.ModuleList(fc)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv1:
            paras = layer(*paras)
        for layer in self.conv2:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class LeNetFeature(nn.Module):
    def __init__(self, input_dim, hidden=512):
        super(LeNetFeature, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 6, 5, bias=False, mean_normalize=True))
        conv1.append(BoundMaxPool2d(2))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(6, 16, 5, bias=False, mean_normalize=True))
        conv2.append(BoundMaxPool2d(2))
        self.conv2 = nn.ModuleList(conv2)

        fc = []
        fc.append(NormDist(16 * 5 * 5, 120, bias=False, mean_normalize=True))
        fc.append(NormDist(120, hidden, bias=False, mean_normalize=True))
        self.fc = nn.ModuleList(fc)
        self.out_features = hidden

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv1:
            paras = layer(*paras)
        for layer in self.conv2:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        return paras
