from __future__ import absolute_import
from __future__ import print_function

import os

import PIL
import numpy as np
import os.path as osp
import shutil
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from .iotools import mkdir_if_missing


def vis_result(distmat, dataset, save_dir=None, topk=10):
    query, gallery = dataset
    num_q, num_g = distmat.shape
    assert num_q == len(query)
    assert num_g == len(gallery)
    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))
    mkdir_if_missing(save_dir)

    fig, axes = plt.subplots(1, topk + 1, figsize=(3 * topk, 6))

    for q_idx in range(num_q):
        query_i = distmat[q_idx]
        indices = np.argsort(query_i)[::-1]  # 余弦相似度
        # indices = np.argsort(query_i)
        qimg_path, qpid, qcamid = query.dataset[q_idx]
        query_name = qimg_path.split("/")[-1]
        query_name = query_name.split("\\")[-1]
        query_img = PIL.Image.open(qimg_path)
        query_img = query_img.resize((256, 256), PIL.Image.ANTIALIAS)
        # query_img = np.rollaxis(np.asarray(query_img, dtype=np.uint8), 0, 3)
        plt.clf()
        ax = fig.add_subplot(1, topk + 1, 1)
        ax.imshow(query_img)
        ax.set_title(f'V_ID:{qpid}')
        ax.axis("off")
        for g_i in range(topk):
            similarity = distmat[q_idx][indices[g_i]]
            ax = fig.add_subplot(1, topk + 1, g_i + 2)
            gimg_path, gpid, gcamid = gallery.dataset[indices[g_i]]
            gallery_img = PIL.Image.open(gimg_path)
            gallery_img = gallery_img.resize((256, 256), PIL.Image.ANTIALIAS)
            # gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
            ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.size[0] - 1,
                                       height=gallery_img.size[1] - 1, edgecolor=(0, 0, 0),
                                       fill=False, linewidth=5))
            ax.imshow(gallery_img)
            ax.set_title(f'{similarity:.3f}/V_ID:{gpid}')
            ax.axis("off")
        plt.tight_layout()
        # 存储重识别可视化结果
        filepath = os.path.join(save_dir, "{}".format(query_name))
        fig.savefig(filepath)
    print("Done")