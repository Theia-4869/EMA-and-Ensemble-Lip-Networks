import torch.nn as nn

from model.norm_dist import NormDistConv, NormDist
from model.bound_module import BoundReLU, BoundMeanNorm, BoundLinear
from model.bound_module import BoundMaxPool2d, BoundAvgPool2d, BoundAdaptiveMaxPool2d, BoundAdaptiveAvgPool2d

class VGGNet(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(VGGNet, self).__init__()
        self.layer1 = self._make_layer(3, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 4)
        self.layer4 = self._make_layer(256, 512, 4)
        self.layer5 = self._make_layer(512, 512, 4, False)

        fc = []
        fc.append(NormDist(512 * 2 * 2, 256, bias=False))
        fc.append(BoundReLU())
        fc.append(NormDist(256, 256, bias=False))
        fc.append(BoundReLU())
        fc.append(NormDist(256, num_classes))
        self.fc = nn.ModuleList(fc)

    def _make_layer(self, in_channel, out_channel, conv_num, pool=True):
        layers = []

        layers.append(NormDistConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, 
                                    stride=1, padding=1, bias=False, mean_normalize=True))
        layers.append(BoundReLU())

        for i in range(conv_num):
            layers.append(NormDistConv(in_channels=out_channel, out_channels=out_channel, kernel_size=3,  
                                        stride=1, padding=1, bias=False, mean_normalize=True))
            layers.append(BoundReLU())
        
        if pool:
            layers.append(BoundMaxPool2d(2))

        return nn.ModuleList(layers)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.layer1:
            paras = layer(*paras)
        for layer in self.layer2:
            paras = layer(*paras)
        for layer in self.layer3:
            paras = layer(*paras)
        for layer in self.layer4:
            paras = layer(*paras)
        for layer in self.layer5:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class VGGNetFeature(nn.Module):
    def __init__(self, input_dim):
        super(VGGNetFeature, self).__init__()
        self.layer1 = self._make_layer(3, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 4)
        self.layer4 = self._make_layer(256, 512, 4)
        self.layer5 = self._make_layer(512, 512, 4, False)

        fc = []
        fc.append(NormDist(512 * 2 * 2, 256, bias=False))
        fc.append(BoundReLU())
        fc.append(NormDist(256, 256, bias=False))
        self.fc = nn.ModuleList(fc)
        self.out_features = 256

    def _make_layer(in_channel, out_channel, conv_num, pool=True):
        layers = []

        layers.append(NormDistConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, 
                                    stride=1, padding=1, bias=False, mean_normalize=True))
        layers.append(BoundReLU())

        for i in range(conv_num):
            layers.append(NormDistConv(in_channels=out_channel, out_channels=out_channel, kernel_size=3,  
                                        stride=1, padding=1, bias=False, mean_normalize=True))
            layers.append(BoundReLU())
        
        if pool:
            layers.append(BoundMaxPool2d(2))

        return nn.ModuleList(layers)

    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.layer1:
            paras = layer(*paras)
        for layer in self.layer2:
            paras = layer(*paras)
        for layer in self.layer3:
            paras = layer(*paras)
        for layer in self.layer4:
            paras = layer(*paras)
        for layer in self.layer5:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        for layer in self.fc:
            paras = layer(*paras)
        return paras
