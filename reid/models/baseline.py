# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import os.path as osp

from reid.models.resnet import ResNet, BasicBlock, Bottleneck
from reid.models.vip import vip_small, vip_medium, vip_base
from reid.models.pvt import pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5

from reid.utils.iotools import check_isfile

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, height, width):
        super(Baseline, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            model_path = osp.join(model_path, "resnet50-19c8e357.pth")
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'vip_small':
            model_path = osp.join(model_path, "vip_s_dict.pth")
            self.in_planes = 768
            self.base = vip_small(in_chans=3, has_last_encoder=False)
        elif model_name == 'vip_medium':
            self.in_planes = 768
            self.base = vip_medium(in_chans=3, has_last_encoder=False)
        elif model_name == 'vip_base':
            self.in_planes = 1024
            self.base = vip_base(in_chans=3, has_last_encoder=False)
        elif model_name == 'pvt_v2_b1':
            self.in_planes = 512
            model_path = osp.join(model_path, "pvt_v2_b1.pth")
            self.base = pvt_v2_b1()
        elif model_name == 'pvt_v2_b2':
            self.in_planes = 512
            model_path = osp.join(model_path, "pvt_v2_b2.pth")
            self.base = pvt_v2_b2()
        elif model_name == 'pvt_v2_b3':
            self.in_planes = 512
            model_path = osp.join(model_path, "pvt_v2_b3.pth")
            self.base = pvt_v2_b3()
        elif model_name == 'pvt_v2_b4':
            self.in_planes = 512
            self.base = pvt_v2_b4()
        elif model_name == 'pvt_v2_b5':
            self.in_planes = 512
            self.base = pvt_v2_b5()


        if model_path and check_isfile(model_path):
            print(f"Load pretrain parameter from: {model_path}")
            self.base.load_param(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        if self.model_name == "hpm" or self.model_name == 'rgap':
            cls_score_list, feat_list = self.base(x)
            if self.training:
                return cls_score_list, feat_list
            else:
                return torch.cat(feat_list, dim=1)

        elif self.model_name == 'rgapn':
            feat1, feat2 = self.base(x)
            if self.training:
                x = feat2
            else:
                x = feat1
        else:
            x = self.base(x)  # 模型提取特征

        if "pvt" in self.model_name:
            global_feat = x
        else:
            global_feat = self.gap(x)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet losses
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
