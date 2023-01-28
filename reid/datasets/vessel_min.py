import random
import os.path as osp
from .base import BaseImageDataset


class Vessel_min(BaseImageDataset):
    dataset_dir = 'vessel_min'  # 民用船

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super(Vessel_min, self).__init__(root)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.txtdir = osp.join(self.dataset_dir, 'annotations')

        self.train_txt = osp.join(self.txtdir, "train.txt")
        self.val_txt = osp.join(self.txtdir, "val.txt")
        self.imgs_dir = osp.join(self.dataset_dir, 'images')

        required_files = [
            self.imgs_dir,
            self.train_txt,
            self.val_txt
        ]
        self._check_before_run(required_files)

        train = self._preprocess(self.train_txt, is_train=True)
        query, gallery = self._preprocess(self.val_txt, is_train=False)

        if verbose:
            print("=> vessel loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self, required_files):
        """Check if all files are available before going deeper"""
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def _preprocess(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        vid_c = {}
        count = 0
        for idx, line in enumerate(img_list_lines):
            line = line.strip()  # 从txt文件读入，单行  图片名  船舶ID
            vid = line.split(" ")[1]
            if vid not in vid_c:
                vid_c[vid] = count
                count += 1
            new_vid = vid_c[vid]  # relabel
            imgid = line.split(' ')[0]
            img_path = osp.join(self.imgs_dir, imgid + ".jpg")
            camid = new_vid
            dataset.append((img_path, new_vid, camid))

        if is_train:
            # train_vid = set()
            # for sample in dataset:
            #     if sample[1] not in train_vid:
            #         train_vid.add(sample[1])
            return dataset
        else:
            random.shuffle(dataset)
            dic_count = {}
            vid_container = set()
            query = []
            gallery = []
            for sample1 in dataset:  # 统计每个id的图片数量
                if sample1[1] not in dic_count.keys():
                    dic_count[sample1[1]] = 1
                else:
                    dic_count[sample1[1]] += 1
            for sample in dataset:  # 仅使用图片数量不少于2的船舶ID
                if dic_count[sample[1]] > 1:
                    if sample[1] not in vid_container:
                        vid_container.add(sample[1])
                        query.append(sample)  # 每个ID仅有一张图片被放入query中
                        # print(sample[0])
                    else:
                        # query.append(sample)
                        gallery.append(sample)
            # query_vid = set()
            # for sample in dataset:
            #     if sample[1] not in query_vid:
            #         query_vid.add(sample[1])
            # gallery_vid = set()
            # for sample in dataset:
            #     if sample[1] not in gallery_vid:
            #         gallery_vid.add(sample[1])
            return query, gallery
