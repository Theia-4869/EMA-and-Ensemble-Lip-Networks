import torch
import argparse
import re
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

from utils import random_seed, create_result_dir, plot_grad_flow, Logger, TableLogger, AverageMeter
from attack import AttackPGD
from adamw import AdamW
from madam import Madam
from model.model import Model, set_eps, get_eps
from model.norm_dist import set_p_norm, get_p_norm

parser = argparse.ArgumentParser(description='Adversarial Robustness')
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--model-list', default='MLPFeature(depth=4,width=4)', type=str) # mlp,conv
parser.add_argument('--predictor-hidden-size', default=512, type=int) # 0 means not to use linear predictor
parser.add_argument('--loss', default='cross_entropy', type=str) # cross_entropy, hinge

parser.add_argument('--p-start', default=8.0, type=float)
parser.add_argument('--p-end', default=1000.0, type=float)
parser.add_argument('--kappa', default=1.0, type=float)
parser.add_argument('--epochs', default='0,50,50,350,400', type=str) # epoch1-epoch3: inc eps; epoch2-epoch4: inc p

parser.add_argument('--eps-test', default=None, type=float)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.99, type=float)
parser.add_argument('--epsilon', default=1e-10, type=float)
parser.add_argument('--wd', default=0.0, type=float)
parser.add_argument('--alpha', type=float, default=0.99) # alpha for ema prototypes update

parser.add_argument('--checkpoint-list', default=None, type=str)

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--dist-url', default='tcp://localhost:23456')
parser.add_argument('--world-size', default=1)
parser.add_argument('--rank', default=0)

parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency')
parser.add_argument('--result-dir', default='ensemble_result/', type=str)
parser.add_argument('--filter-name', default='', type=str)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--visualize', action='store_true')


def cal_acc(outputs, targets):
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean()


def parallel_reduce(*argv):
    tensor = torch.FloatTensor(argv).cuda()
    torch.distributed.all_reduce(tensor)
    ret = tensor.cpu() / torch.distributed.get_world_size()
    return ret.tolist()


@torch.no_grad()
def ensemble_test(net_list, weight_list, loss_fun, testloader, logger, test_logger, gpu, parallel, print_freq):
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    test_loader_len = len(testloader)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = 0
        for net, weight in zip(net_list, weight_list):
            outputs += weight * net(inputs)
        loss = loss_fun(outputs, targets)
        losses.update(loss.mean().item(), targets.size(0))
        accs.update(cal_acc(outputs, targets).item(), targets.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                             batch_idx + 1, test_loader_len, batch_time=batch_time, loss=losses, acc=accs))

    loss, acc = losses.avg, accs.avg
    if parallel:
        loss, acc = parallel_reduce(losses.avg, accs.avg)
    if test_logger is not None:
        test_logger.log({'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('test loss  ' + f'{loss:.4f}' + '   acc ' + f'{acc:.4f}')
    return loss, acc


def ensemble_gen_adv_examples(model_list, weight_list, attacker_list, test_loader, gpu, parallel, logger, fast=False):
    correct = 0
    tot_num = 0
    size = len(test_loader)
    net_list = model_list
    tot_acc = 0
    for model, attacker in zip(model_list, attacker_list):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            result = torch.ones(targets.size(
                0), dtype=torch.bool, device=targets.device)
            for i in range(1):
                perturb = attacker.find(inputs, targets)
                with torch.no_grad():
                    outputs = 0
                    for net, weight in zip(net_list, weight_list):
                        outputs += weight * net(perturb)
                    predicted = torch.max(outputs.data, 1)[1]
                    result &= (predicted == targets)
            correct += result.float().sum().item()
            tot_num += inputs.size(0)
            if fast and batch_idx * 10 >= size:
                break

        acc = correct / tot_num * 100
        if parallel:
            acc, = parallel_reduce(acc)
        tot_acc = max(tot_acc, acc)
        
    
    if logger is not None:
            logger.print('adversarial attack acc ' + f'{tot_acc:.4f}')
    return tot_acc


@torch.no_grad()
def ensemble_certified_test(net_list, weight_list, eps, up, down, testloader, logger, gpu, parallel):
    tot_outputs = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        lower = torch.max(inputs - eps, down)
        upper = torch.min(inputs + eps, up)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = 0
        for net, weight in zip(net_list, weight_list):
            outputs += weight * net(inputs, lower=lower, upper=upper, targets=targets)
        tot_outputs.append(outputs[1])
        labels.append(targets)
    tot_outputs = torch.cat(tot_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    res = (tot_outputs.max(dim=1)[1] == labels).float().mean().item()

    if parallel:
        res, = parallel_reduce(res)
    if logger is not None:
        logger.print(' certified acc ' + f'{res:.4f}')
    return res


def parse_function_call(s):
    s = re.split(r'[()]', s)
    if len(s) == 1:
        return s[0], {}
    name, params, _ = s
    params = re.split(r',\s*', params)
    params = dict([p.split('=') for p in params])
    for key, value in params.items():
        try:
            params[key] = int(params[key])
        except ValueError:
            try:
                params[key] = float(params[key])
            except ValueError:
                pass
    return name, params


import torch.nn.functional as F
def cross_entropy():
    return F.cross_entropy


# The hinge loss function is a combination of max_hinge_loss and average_hinge_loss.
def hinge(mix=0.75):
    def loss_fun(outputs, targets):
        return mix * outputs.max(dim=1)[0].clamp(min=0).mean() + (1 - mix) * outputs.clamp(min=0).mean()
    return loss_fun


class Loss():
    def __init__(self, loss, kappa):
        self.loss = loss
        self.kappa = kappa

    def __call__(self, *args):
        margin_output = args[0] - torch.gather(args[0], dim=1, index=args[-1].view(-1, 1))
        if len(args) == 2:
            return self.loss(margin_output, args[-1])
        # args[1] which corresponds to worse_outputs, is already a margin vector.
        return self.kappa * self.loss(args[1], args[-1]) + (1 - self.kappa) * self.loss(margin_output, args[-1])


def main_worker(gpu, parallel, args, result_dir):
    if parallel:
        args.rank = args.rank + gpu
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    torch.backends.cudnn.benchmark = True
    random_seed(args.seed + args.rank) # make data aug different for different processes
    torch.cuda.set_device(gpu)

    assert args.batch_size % args.world_size == 0
    from dataset import load_data, get_statistics, default_eps, input_dim
    train_loader, test_loader = load_data(args.dataset, 'data/', args.batch_size // args.world_size, parallel,
                                          augmentation=True, classes=None)
    mean, std = get_statistics(args.dataset)
    num_classes = len(train_loader.dataset.classes)

    from model.bound_module import Predictor, BoundFinalIdentity
    from model.mlp import MLPFeature, MLP
    from model.conv import ConvFeature, Conv
    from model.lenet import LeNetFeature, LeNet
    from model.alexnet import AlexNetFeature, AlexNet
    from model.alexnet2 import AlexNetFeature2, AlexNet2
    from model.vggnet import VGGNetFeature, VGGNet
    from model.resnet import ResNetFeature, ResNet
    args.model_list = re.split(r',\s', args.model_list[1:-1])
    args.model_num = len(args.model_list)
    model_list = []
    weight_list = []
    for m in args.model_list:
        model_name, params = parse_function_call(m)
        if args.predictor_hidden_size > 0:
            model = locals()[model_name](input_dim=input_dim[args.dataset], **params)
            predictor = Predictor(model.out_features, args.predictor_hidden_size, num_classes)
        else:
            model = locals()[model_name](input_dim=input_dim[args.dataset], num_classes=num_classes, **params)
            predictor = BoundFinalIdentity()
        model = Model(model, predictor, eps=0)
        model = model.cuda(gpu)
        if parallel:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_list.append(model)
        weight_list.append(float(1 / args.model_num))

    loss_name, params = parse_function_call(args.loss)
    loss = Loss(globals()[loss_name](**params), args.kappa)

    output_flag = not parallel or gpu == 0
    if output_flag:
        logger = Logger(os.path.join(result_dir, 'log.txt'))
        for arg in vars(args):
            logger.print(arg, '=', getattr(args, arg))
        logger.print(train_loader.dataset.transform)
        logger.print(model)
        logger.print('number of params: ', sum([p.numel() for p in model.parameters()]))
        logger.print('Using loss', loss)
        test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['loss', 'acc'])
    else:
        logger = test_logger = None

    if args.checkpoint_list:
        args.checkpoint_list = re.split(r',\s', args.checkpoint_list[1:-1])
        for c, m in zip(args.checkpoint_list, model_list):
            assert os.path.isfile(c)
            if parallel:
                torch.distributed.barrier()
            checkpoint = torch.load(c, map_location=lambda storage, loc: storage.cuda(gpu))
            state_dict = checkpoint['state_dict']
            if next(iter(state_dict))[0:7] == 'module.' and not parallel:
                new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                state_dict = new_state_dict
            elif next(iter(state_dict))[0:7] != 'module.' and parallel:
                new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                state_dict = new_state_dict
            m.load_state_dict(state_dict)
            print("=> loaded '{}'".format(args.checkpoint))
            if parallel:
                torch.distributed.barrier()

    if args.eps_test is None:
        args.eps_test = default_eps[args.dataset]
    args.eps_test /= std
    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).cuda(gpu)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).cuda(gpu)
    attacker_list = []
    for m in model_list:
        attacker = AttackPGD(m, args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
        attacker_list.append(attacker)

    if args.visualize and output_flag:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(result_dir)
    else: 
        writer = None
    
    if logger is not None:
        logger.print("Calculate test accuracy on test dataset")
    test_loss, test_acc = ensemble_test(model_list, weight_list, loss, test_loader, logger, test_logger, gpu, parallel, args.print_freq)
    writer.add_scalar('curve/p', get_p_norm(model), 0)
    writer.add_scalar('curve/test loss', test_loss, 0)
    writer.add_scalar('curve/test acc', test_acc, 0)

    if logger is not None:
        logger.print("Generate adversarial examples on test dataset")
    robust_test_acc = ensemble_gen_adv_examples(model_list, attacker_list, test_loader, gpu, parallel, logger)
    if writer is not None:
        writer.add_scalar('curve/robust test acc', robust_test_acc, 0)
    
    if logger is not None:
        logger.print("Calculate certified accuracy on test dataset")
    certified_acc = ensemble_certified_test(model, args.eps_test, up, down, test_loader, logger, gpu, parallel)
    if writer is not None:
        writer.add_scalar('curve/certified acc', certified_acc, 0)

    if writer is not None:
        writer.close()

def main(father_handle, **extra_argv):
    args = parser.parse_args()
    for key, val in extra_argv.items():
        setattr(args, key, val)
    result_dir = create_result_dir(args)
    if father_handle is not None:
        father_handle.put(result_dir)
    if args.gpu != -1:
        main_worker(args.gpu, False, args, result_dir)
    else:
        n_procs = torch.cuda.device_count()
        args.world_size *= n_procs
        args.rank *= n_procs
        torch.multiprocessing.spawn(main_worker, nprocs=n_procs, args=(True, args, result_dir))


if __name__ == '__main__':
    main(None)
