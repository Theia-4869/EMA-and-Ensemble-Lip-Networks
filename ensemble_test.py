def ensemble_test(net_list, weigh_list, loss_fun, epoch, testloader, logger, test_logger, gpu, parallel, print_freq):
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    test_loader_len = len(testloader)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = 0
        for net , weight in zip(net_list, weigh_list):
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
        test_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('Epoch %d:  ' % epoch + 'test loss  ' +
                     f'{loss:.4f}' + '   acc ' + f'{acc:.4f}')
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