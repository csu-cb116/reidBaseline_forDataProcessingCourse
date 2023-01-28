import argparse

import yaml


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=r'./configs/test.yml',
                        help='resnet.yml or vip.yml or pvt.yml or test.yml')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default=r'/home/xyc/datasets/',
                        help='12:/home/project/xyc/datasets/ 184:/home/xyc/datasets/  2080ti: D:/Dataset'
                             '3080ti: /mnt/data/xyc/datasets/')
    parser.add_argument('--dataset_name', type=str, default='vessel_min',
                        help='source dataset for training(delimited by space)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    # split-id not used
    parser.add_argument('--split-id', type=int, default=0,
                        help='split index (note: 0-based)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=256,
                        help='width of an image')
    parser.add_argument('--train-sampler', type=str, default='RandomSampler',
                        help='sampler for trainloader')

    # ************************************************************
    # Data augmentation
    # ************************************************************
    parser.add_argument('--random-erase', action='store_true',
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', action='store_true',
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', action='store_true',
                        help='randomly alter the intensities of RGB channels')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adamw',
                        help='optimization algorithm (optimizers.py)')
    parser.add_argument('--lr', default=0.0003, type=float,
                        help='initial learning rate, default:0.0003 vip_base:0.01 adam:0.0003 sgd:0.1')
    parser.add_argument('--weight-decay', default=0.05, type=float,
                        help='weight decay, default=5e-04, vip_base:0.1, pvt:0.05')
    parser.add_argument('--staged_lr', default=True, type=bool,
                        help='different lr for different layers')
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help='exponential decay rate for adam\'s second moment')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=120, type=int,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')

    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=32, type=int,
                        help='test batch size')

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[40, 70], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')
    parser.add_argument('--warmup_iters', default=5, type=int,
                        help='the number of warmup epoch')
    parser.add_argument('--warmup_factor', default=0.01, type=float,
                        help='warm up factor')
    parser.add_argument('--warmup_method', default='linear', type=str,
                        help='method of warm up, option: constant, linear')

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help='use label smoothing regularizer in cross entropy loss')

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--lambda-xent', type=float, default=1,
                        help='weight to balance cross entropy loss')
    parser.add_argument('--lambda-htri', type=float, default=1,
                        help='weight to balance hard triplet loss')

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='vip_small',
                        help='resnet50 vip_small vip_medium pvt_v2_b1 pvt_v2_b2 pvt_v2_b3 pvt_v2_b4 pvt_v2_b5 pvt_v2_b6 ')
    parser.add_argument('--last_stride', type=int, default=2)
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')
    parser.add_argument('--neck', type=str, default='bnneck',
                        help='no or bnneck')
    parser.add_argument('--neck_feat', type=str, default='after',
                        help='test with feature before or after BN')
    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='./pretrain/',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')
    parser.add_argument('--eval-freq', type=int, default=5,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--test_size', type=int, default=800,
                        help='test-size for vehicleID dataset, choices=[800,1600,2400]')
    parser.add_argument('--query-remove', type=bool, default=True)
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=1,
                        help='manual seed')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=r"D:\Pycharm_Projects\ViP_VReID_base_temp\log\train\vessel_jun\vip_small_20221214-1706\checkpoints\epoch_10.pth",
                        help='checkpoint for testing')
    parser.add_argument('--save-dir', type=str, default='./log',
                        help='path to save log and model weights')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--visualize-ranks', action='store_true',
                        help='visualize ranked results, only available in evaluation mode')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    return parser



def dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'dataset_name': parsed_args.dataset_name,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
        'staged_lr': parsed_args.staged_lr,
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
        'warmup_iters': parsed_args.warmup_iters,
        'warmup_factor': parsed_args.warmup_factor,
        'warmup_method': parsed_args.warmup_method,
    }
