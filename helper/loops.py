from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
import random
import copy

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def predictions(x):
    _, preds = torch.max(x, 1)
    return preds

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def crd_mixup_data(x, y, indices, contrast_indices, net_kd, net, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    preact = False
    feat_a_s, logit_a_s = net_kd(x.cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_a_t, logit_a_t = net(x.cuda(), is_feat=True, preact=preact)
    
    feat_b_s, logit_b_s = net_kd(x[index,:].cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_b_t, logit_b_t = net(x[index,:].cuda(), is_feat=True, preact=preact)

    y_a_t, y_b_t = predictions(logit_a_t), predictions(logit_b_t)
    index_a_t, index_b_t = indices, indices[index]
    con_index_a_t, con_index_b_t = contrast_indices, contrast_indices[index]
    # y_a_t, y_b_t = predictions(net(x.to(device))), predictions(net(x[index, :].to(device)))
    # z_a_t, z_b_t = net(x.to(device)), net(x[index, :].to(device))
    # z_a_s, z_b_s = net_kd(x.to(device)), net_kd(x[index, :].to(device))
    return mixed_x, y_a_t, y_b_t, index_a_t, index_b_t, con_index_a_t, con_index_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam

def mixup_data(x, y, net_kd, net, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    preact = False
    feat_a_s, logit_a_s = net_kd(x.cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_a_t, logit_a_t = net(x.cuda(), is_feat=True, preact=preact)
    
    feat_b_s, logit_b_s = net_kd(x[index,:].cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_b_t, logit_b_t = net(x[index,:].cuda(), is_feat=True, preact=preact)

    y_a_t, y_b_t = predictions(logit_a_t), predictions(logit_b_t)
    return mixed_x, y_a_t, y_b_t, logit_a_t, logit_b_t, logit_a_s, logit_b_s, lam

def rkd_mixup_data(x, y, net_kd, net, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    preact = False
    feat_a_s, logit_a_s = net_kd(x.cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_a_t, logit_a_t = net(x.cuda(), is_feat=True, preact=preact)
    
    feat_b_s, logit_b_s = net_kd(x[index,:].cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_b_t, logit_b_t = net(x[index,:].cuda(), is_feat=True, preact=preact)

    y_a_t, y_b_t = predictions(logit_a_t), predictions(logit_b_t)
    return mixed_x, y_a_t, y_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rkd_cutmix_data(x, y, net_kd, net, beta=1., use_cuda=True):

    lam = np.random.beta(beta, beta)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    cutmix_x = copy.deepcopy(x)
    bbx1, bby1, bbx2, bby2 = rand_bbox(cutmix_x.size(), lam)
    cutmix_x[:, bbx1:bbx2, bby1:bby2] = x[index, :][:, bbx1:bbx2, bby1:bby2]
    lam = 1-((bbx2 - bbx1)*(bby2 - bby1)/(cutmix_x.size()[-1]*cutmix_x.size()[-2]))

    preact=False
    feat_a_s, logit_a_s = net_kd(x.cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_a_t, logit_a_t = net(x.cuda(), is_feat=True, preact=preact)
    
    feat_b_s, logit_b_s = net_kd(x[index,:].cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_b_t, logit_b_t = net(x[index,:].cuda(), is_feat=True, preact=preact)

    y_a_t, y_b_t = predictions(logit_a_t), predictions(logit_b_t)
    return cutmix_x, y_a_t, y_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam

def crd_cutmix_data(x, y, indices, contrast_indices, net_kd, net, beta=1., use_cuda=True):
    
    lam = np.random.beta(beta, beta)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    cutmix_x = copy.deepcopy(x)
    bbx1, bby1, bbx2, bby2 = rand_bbox(cutmix_x.size(), lam)
    cutmix_x[:, bbx1:bbx2, bby1:bby2] = x[index, :][:, bbx1:bbx2, bby1:bby2]
    lam = 1-((bbx2 - bbx1)*(bby2 - bby1)/(cutmix_x.size()[-1]*cutmix_x.size()[-2]))
    
    preact = False
    feat_a_s, logit_a_s = net_kd(x.cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_a_t, logit_a_t = net(x.cuda(), is_feat=True, preact=preact)
    
    feat_b_s, logit_b_s = net_kd(x[index,:].cuda(), is_feat=True, preact=preact)
    with torch.no_grad():
        feat_b_t, logit_b_t = net(x[index,:].cuda(), is_feat=True, preact=preact)

    y_a_t, y_b_t = predictions(logit_a_t), predictions(logit_b_t)
    index_a_t, index_b_t = indices, indices[index]
    con_index_a_t, con_index_b_t = contrast_indices, contrast_indices[index]
    return cutmix_x, y_a_t, y_b_t, index_a_t, index_b_t, con_index_a_t, con_index_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        elif opt.distill in ['mcrd']:
            input, target, index, contrast_idx = data
            input, targets_a, targets_b, index_a_t, index_b_t, con_index_a_t, con_index_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam = crd_mixup_data(input, target, index, contrast_idx, model_s, model_t)
        elif opt.distill in ['cmcrd']:
            r = np.random.rand(1)
            input, target, index, contrast_idx = data
            if r>0.5:
                flag=0
                with torch.no_grad():
                    _, target = torch.max(model_t(input.cuda()),1)
            else:
                flag=1
                input, targets_a, targets_b, index_a_t, index_b_t, con_index_a_t, con_index_b_t, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam = crd_cutmix_data(input, target, index, contrast_idx, model_s, model_t)
        else:
            input, target, index = data
            if opt.distill in ['mkd']:
                input, targets_a, targets_b, logit_a_t, logit_b_t, logit_a_s, logit_b_s, lam = mixup_data(input, target, model_s, model_t)
            if opt.distill in ['mrkd']:
                input, targets_a, targets_b, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam = rkd_mixup_data(input, target, model_s, model_t)
            if opt.distill in ['cmrkd']:
                r = np.random.rand(1)
                if r>0.5:
                    flag=0
                    with torch.no_grad():
                        _, target = torch.max(model_t(input.cuda()),1)
                else:
                    flag=1
                    input, targets_a, targets_b, feat_a_t, feat_b_t, feat_a_s, feat_b_s, lam = rkd_cutmix_data(input, target, model_s, model_t)
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        if opt.distill in ['mcrd', 'mkd', 'mrkd']:
            loss_cls = mixup_criterion(criterion_cls, logit_s, targets_a, targets_b, lam)
        elif opt.distill in ['cmrkd', 'cmcrd']:
            if flag==1:
                loss_cls = mixup_criterion(criterion_cls, logit_s, targets_a, targets_b, lam)
            else:
                loss_cls = criterion_cls(logit_s, target)
        else:
            loss_cls = criterion_cls(logit_s, target)
        if opt.distill in ['mkd']:
            loss_div = criterion_div(logit_a_s, logit_a_t)+criterion_div(logit_b_s, logit_b_t)
        else:
            loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd' or opt.distill == 'mkd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd' or (opt.distill == 'cmcrd' and flag==0):
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index.cuda(), contrast_idx.cuda())
        elif opt.distill == 'cmcrd' and flag==1:
            f_a_s = feat_a_s[-1]
            f_a_t = feat_a_t[-1]
            f_b_s = feat_b_s[-1]
            f_b_t = feat_b_t[-1]
            loss_kd = criterion_kd(f_a_s, f_a_t, index_a_t.cuda(), con_index_a_t.cuda()) + criterion_kd(f_b_s, f_b_t, index_b_t.cuda(), con_index_b_t.cuda())
        elif opt.distill == 'mcrd':
            f_a_s = feat_a_s[-1]
            f_a_t = feat_a_t[-1]
            f_b_s = feat_b_s[-1]
            f_b_t = feat_b_t[-1]
            loss_kd = criterion_kd(f_a_s, f_a_t, index_a_t.cuda(), con_index_a_t.cuda()) + criterion_kd(f_b_s, f_b_t, index_b_t.cuda(), con_index_b_t.cuda())
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd' or (opt.distill=='cmrkd' and flag==0):
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'cmrkd' and flag==1:
            f_a_s = feat_a_s[-1]
            f_a_t = feat_a_t[-1]
            f_b_s = feat_b_s[-1]
            f_b_t = feat_b_t[-1]
            loss_kd = criterion_kd(f_a_s, f_a_t)+criterion_kd(f_b_s, f_b_t)
        elif opt.distill == 'mrkd':
            f_a_s = feat_a_s[-1]
            f_a_t = feat_a_t[-1]
            f_b_s = feat_b_s[-1]
            f_b_t = feat_b_t[-1]
            loss_kd = criterion_kd(f_a_s, f_a_t)+criterion_kd(f_b_s, f_b_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill in ['cmcrd', 'cmrkd']:
            loss = opt.gamma * loss_cls + opt.beta * loss_kd
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, opt, num_bins=100):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(val_loader.dataset))
    counts = [0 for i in range(num_bins+1)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            probabilities = torch.nn.functional.softmax(output, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, target):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
            
    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))
    print(' * ECE {ece:.4f} OE {oe:.4f}'
              .format(ece=ECE, oe=OE))

    return top1.avg, top5.avg, losses.avg, ECE, OE
