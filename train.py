from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from reid.data_manager import ImageDataManager
from reid import models
from reid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from reid.utils.iotools import check_isfile
from reid.utils.avgmeter import AverageMeter
from reid.utils.loggers import Logger, RankLogger
from reid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from reid.utils.generaltools import set_random_seed
from reid.eval_metrics import evaluate
from reid.optimizers import init_optimizer
from reid.lr_schedulers import init_lr_scheduler, WarmupMultiStepLR
from tensorboardX import SummaryWriter

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    args.save_dir = osp.join(args.save_dir, "train", args.dataset_name, args.arch + "_" + time_now)
    checkpoint_save_dir = osp.join(args.save_dir, 'checkpoints')
    logdir = osp.join(args.save_dir, log_name)
    tbdir = osp.join(args.save_dir, 'tensorboard')
    writer = SummaryWriter(tbdir)
    sys.stdout = Logger(logdir)

    # writer = SummaryWriter(args.save_dir)
    # logdir = osp.join(args.save_dir, log_name)
    # sys.stdout = Logger(logdir)
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, queryloader, galleryloader = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))

    model = models.Baseline(num_classes=dm.num_train_pids, last_stride=args.last_stride, model_path=args.load_weights,
                            neck=args.neck, neck_feat=args.neck_feat, model_name=args.arch, height=args.height,
                            width=args.width)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = init_optimizer(model, args.arch, **optimizer_kwargs(args))
    # scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
    scheduler = WarmupMultiStepLR(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.evaluate:
        print('Evaluate only')

        print('Evaluating {} ...'.format(args.dataset_name))
        distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

        if args.visualize_ranks:
            visualize_ranked_results(
                distmat, (queryloader, galleryloader),
                save_dir=osp.join(checkpoint_save_dir, 'ranked_results', args.dataset_name),
                topk=20
            )

    time_start = time.time()
    ranklogger = RankLogger(args.dataset_name)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    best_rank = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, writer)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')

            print('Evaluating {} ...'.format(args.dataset_name))
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            ranklogger.write(args.dataset_name, epoch + 1, rank1)
            if best_rank < rank1:
                best_rank = rank1
                is_best = True
            else:
                is_best = False

            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': args.arch,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_save_dir, is_best=is_best)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, writer):
    losses = AverageMeter()
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True  # open all layers

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)

        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])
        current_lr = optimizer.param_groups[0]['lr']
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: {3}\t'
                  'batchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'totalLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'idLoss {xent.val:.4f} ({xent.avg:.4f})\t'
                  'triLoss {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                current_lr,
                batch_time=batch_time,
                loss=losses,
                xent=xent_losses,
                htri=htri_losses,
                acc=accs
            ))
            niter = epoch * len(trainloader) + batch_idx
            writer.add_scalar('totalLoss', losses.val, niter)
            writer.add_scalar('idLoss', xent_losses.val, niter)
            writer.add_scalar('triLoss', htri_losses.val, niter)
            writer.add_scalar('Acc', accs.val, niter)

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    # distmat = distmat.numpy()
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    # dist_m = 1 - torch.mm(qf, gf.t())
    dist_m = torch.mm(qf, gf.t())
    distmat = dist_m.cpu().numpy()  # 余弦相似度
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
