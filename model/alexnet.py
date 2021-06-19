import torch.nn as nn

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundTanh, BoundMeanNorm, BoundLinear, BoundFinalLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d

class AlexNet(nn.Module):
    def __init__(self, input_dim, hidden=512, num_classes=10):
        super(AlexNet, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 96, 7, 2, 2, bias=False, mean_normalize=True))
        conv1.append(BoundMaxPool2d(3, 2, 0))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(96, 256, 5, 1, 2, bias=False, mean_normalize=True))
        conv2.append(BoundMaxPool2d(3, 2, 0))
        self.conv2 = nn.ModuleList(conv2)

        conv3 = []
        conv3.append(NormDistConv(256, 384, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv3 = nn.ModuleList(conv3)

        conv4 = []
        conv4.append(NormDistConv(384, 384, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv4 = nn.ModuleList(conv4)

        conv5 = []
        conv5.append(NormDistConv(384, 256, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv5 = nn.ModuleList(conv5)

        fc = []
        fc.append(NormDist(256 * 3 * 3, 1024, bias=False, mean_normalize=True))
        fc.append(NormDist(1024, hidden, bias=False, mean_normalize=True))
        fc.append(NormDist(hidden, num_classes, bias=True, mean_normalize=False))
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
    def __init__(self, input_dim, hidden=512):
        super(AlexNetFeature, self).__init__()
        conv1 = []
        conv1.append(NormDistConv(3, 96, 7, 2, 2, bias=False, mean_normalize=True))
        conv1.append(BoundMaxPool2d(3, 2, 0))
        self.conv1 = nn.ModuleList(conv1)

        conv2 = []
        conv2.append(NormDistConv(96, 256, 5, 1, 2, bias=False, mean_normalize=True))
        conv2.append(BoundMaxPool2d(3, 2, 0))
        self.conv2 = nn.ModuleList(conv2)

        conv3 = []
        conv3.append(NormDistConv(256, 384, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv3 = nn.ModuleList(conv3)

        conv4 = []
        conv4.append(NormDistConv(384, 384, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv4 = nn.ModuleList(conv4)

        conv5 = []
        conv5.append(NormDistConv(384, 256, 3, 1, 1, bias=False, mean_normalize=True))
        self.conv5 = nn.ModuleList(conv5)

        fc = []
        fc.append(NormDist(256 * 3 * 3, 1024, bias=False, mean_normalize=True))
        fc.append(NormDist(1024, hidden, bias=False, mean_normalize=True))
        self.fc = nn.ModuleList(fc)
        self.out_features = hidden

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
