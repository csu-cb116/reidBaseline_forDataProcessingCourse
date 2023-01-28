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

from args import argument_parser, dataset_kwargs
from reid.data_manager import testImageDataManager
from reid import models
from reid.utils.iotools import check_isfile
from reid.utils.avgmeter import AverageMeter
from reid.utils.loggers import Logger
from reid.utils.torchtools import count_num_param, resume_from_checkpoint
from reid.utils.visualtools import vis_result
from reid.utils.generaltools import set_random_seed
from reid.eval_metrics import evaluate

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt'
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    args.save_dir = osp.join(args.save_dir, "test", args.dataset_name, args.arch + "_" + time_now)
    logdir = osp.join(args.save_dir, log_name)
    sys.stdout = Logger(logdir)

    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = testImageDataManager(use_gpu, **dataset_kwargs(args))
    queryloader, galleryloader = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))

    model = models.Baseline(num_classes=1024, last_stride=args.last_stride, model_path=args.load_weights,
                            neck=args.neck, neck_feat=args.neck_feat, model_name=args.arch, height=args.height,
                            width=args.width)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    model = nn.DataParallel(model).cuda() if use_gpu else model

    if args.checkpoint_dir and check_isfile(args.checkpoint_dir):
        print('Loading from {}'.format(args.checkpoint_dir))
        args.start_epoch = resume_from_checkpoint(args.checkpoint_dir, model, is_eval=True)

    time_start = time.time()

    print('=> Test')
    print('Evaluating {} ...'.format(args.dataset_name))

    distmat = test(model, queryloader, galleryloader, use_gpu)

    # vis_result(
    #     distmat, (queryloader.dataset, galleryloader.dataset),
    #     save_dir=osp.join(args.save_dir, 'ranked_results', args.dataset_name),
    #     topk=5
    # )
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
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

    # m, n = qf.size(0), gf.size(0)
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
    with open(os.path.join(args.save_dir, "Rank.txt"), "a") as f:
        f.write("Rank1: " + str(cmc[0]) + "\n")
    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    return distmat


if __name__ == '__main__':
    main()
