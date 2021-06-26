import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, feature, predictor, eps):
        super(Model, self).__init__()
        self.feature = feature
        self.predictor = predictor
        self.eps = eps
    def forward(self, x, lower=None, upper=None, targets=None):
        if targets is None:
            lower = upper = None
        if self.feature is not None:
            x, lower, upper = self.feature(x, lower=lower, upper=upper)
        if targets is not None and (lower is None or upper is None):
            lower = x - self.eps
            upper = x + self.eps
        return self.predictor(x, lower, upper, targets=targets)


class FusionModel(nn.Module):
    def __init__(self, gpu, model_list, num_classes=10):
        super(FusionModel, self).__init__()
        self.gpu = gpu
        self.model_list = nn.ModuleList(model_list)
        self.model_num = len(model_list)
        self.num_class = num_classes
    def forward(self, x, lower=None, upper=None, targets=None):
        batch_size = x.size()[0]
        y = torch.zeros(batch_size, self.num_class).cuda(self.gpu, non_blocking=True)
        res = torch.zeros(batch_size, self.num_class).cuda(self.gpu, non_blocking=True)

        for model in self.model_list:
            output = model(x, lower, upper, targets)
            if targets is None:
                y += output
            else:
                y += output[0]
                res += output[1]
        y /= self.model_num
        res /= self.model_num

        if targets is None:
            return y
        return y, res


class VotingModel(nn.Module):
    def __init__(self, gpu, model_list, num_classes=10):
        super(VotingModel, self).__init__()
        self.gpu = gpu
        self.model_list = nn.ModuleList(model_list)
        self.model_num = len(model_list)
        self.num_class = num_classes
    def forward(self, x, lower=None, upper=None, targets=None):
        batch_size = x.size()[0]
        y = torch.zeros(batch_size, self.num_class).cuda(self.gpu, non_blocking=True)
        res = torch.zeros(batch_size, self.num_class).cuda(self.gpu, non_blocking=True)

        for model in self.model_list:
            output = model(x, lower, upper, targets)
            if targets is None:
                y += F.softmax(output, dim=1)
            else:
                y += F.softmax(output[0], dim=1)
                res += F.softmax(output[1], dim=1)
        y /= self.model_num
        res /= self.model_num

        if targets is None:
            return y
        return y, res


def set_eps(model, eps):
    for m in model.modules():
        if isinstance(m, Model):
            m.eps = eps

def get_eps(model):
    for m in model.modules():
        if isinstance(m, Model):
            return m.eps
    return None