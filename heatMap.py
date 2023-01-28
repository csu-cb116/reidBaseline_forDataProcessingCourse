import os
import sys
import time
import warnings

import torch
from torch.backends import cudnn

from args import argument_parser, dataset_kwargs
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import matplotlib
import os
import os.path as osp

from reid import models
from reid.data_manager import ImageDataManager
from reid.utils.generaltools import set_random_seed
from reid.utils.iotools import check_isfile
from reid.utils.loggers import Logger
from reid.utils.torchtools import resume_from_checkpoint

matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argument_parser()
args = parser.parse_args()


def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, :, :].reshape(tensor.size(0),
                                     height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def getHeatMap(model, target_layers, image_path, save_dir, is_transformer=False):
    for img in os.listdir(image_path):
        imgdir = osp.join(image_path, img)
        heatMapName = img.split(".")[0]
        rgb_img = cv2.imread(imgdir, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (256, 256))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        # Construct the CAM object once, and then re-use it on many images:
        if is_transformer:
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
        else:
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        # targets = [ClassifierOutputTarget(122)]
        targets = None

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, ax = plt.subplots(1, 1)
        ax.set_title(heatMapName, fontsize=12, color='r')
        plt.imshow(visualization, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        savepath = osp.join(save_dir, "images")
        if not osp.exists(savepath):
            os.makedirs(savepath)
        filepath = os.path.join(savepath, heatMapName)
        plt.savefig(filepath)



def main():
    set_random_seed(args.seed)
    args.arch = "vip_small"
    imgdir = r"/home/xyc/Vip_Vreid_base_temp/cam_data/"
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt'
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    args.save_dir = osp.join(args.save_dir, "heatMap", args.arch + "_cam_" + time_now)
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')


    modelpath = "/home/xyc/Vip_Vreid_base_temp/log/vip_small_rank1_76.9/model_best.pth"
    model = models.Baseline(num_classes=1024, last_stride=args.last_stride, model_path=args.load_weights,
                            neck=args.neck, neck_feat=args.neck_feat, model_name=args.arch, height=args.height,
                            width=args.width)
    target_layer = [model.base.last_norm]

    # modelpath = "/home/xyc/Vip_Vreid_base_temp/log/resnet50_rank1_79.1/model_best.pth"
    # model = models.Baseline(num_classes=1024, last_stride=args.last_stride, model_path=args.load_weights,
    #                         neck=args.neck, neck_feat=args.neck_feat, model_name=args.arch, height=args.height,
    #                         width=args.width)
    # target_layer = [model.base.layer3]

    # modelpath = r"D:\Pycharm_Projects\ViP_VReID_base_temp\log\train\vessel_jun\pvt_v2_b2_20221206-2202\checkpoints\epoch_115.pth"
    # model = models.Baseline(num_classes=1024, last_stride=args.last_stride, model_path=args.load_weights,
    #                         neck=args.neck, neck_feat=args.neck_feat, model_name=args.arch, height=args.height,
    #                         width=args.width)
    # target_layer = [model.base.norm4]


    args.resume = modelpath
    if args.resume and check_isfile(args.resume):
        print("Loading the model from " + modelpath)
        model.load_state_dict(torch.load(args.resume), strict=False)
    getHeatMap(model, target_layer, imgdir, args.save_dir, is_transformer=False)
    print("Processed!")


if __name__ == "__main__":
    main()
