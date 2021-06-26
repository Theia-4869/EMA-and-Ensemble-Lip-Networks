import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_result_dir(args):
    result_dir = args.result_dir
    mp = {
        'dataset': '',
        'model': '',
        'predictor_hidden_size': 'hid',
        'loss': None,
        'p_start': 'p',
        'p_end': 'p_end',
        'batch_size': 'bs',
        'beta': None,
        'beta2': None,
        'epsilon': None,
        'start_epoch': None,
        'checkpoint': None,
        'gpu': None,
        'dist_url': None,
        'world_size': None,
        'rank': None,
        'print_freq': None,
        'result_dir': None,
        'filter_name': '',
        'seed': '',
        'visualize': None,
    }
    for arg in vars(args):
        if arg in mp and mp[arg] is None:
            continue
        value = getattr(args, arg)
        if type(value) == bool:
            value = 'T' if value else 'F'
        if type(value) == list:
            value = str(value).replace(' ', '')
        name = mp.get(arg, arg)
        result_dir += name + str(value) + '_'
    return result_dir

def create_result_dir(args):
    result_dir = get_result_dir(args)
    id = 0
    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id

def create_result_dir0(args):
    result_dir = args.result_dir
    id = 0
    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

class Logger(object):
    def __init__(self, dir):
        self.fp = open(dir, 'w')
    def __del__(self):
        self.fp.close()
    def print(self, *args, **kwargs):
        print(*args, file=self.fp, **kwargs)
        print(*args, **kwargs)

class TableLogger(object):
    def __init__(self, path, header):
        import csv
        self.fp = open(path, 'w')
        self.logger = csv.writer(self.fp, delimiter='\t')
        self.logger.writerow(header)
        self.header = header
    def __del__(self):
        self.fp.close()
    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.fp.flush()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EMA():
    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.shadow = {}
        self.backup = {}
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, epoch):
        alpha = min(1 - 1 / (epoch + 1), self.alpha)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - alpha) * param.data + alpha * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

