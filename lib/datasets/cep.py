# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

from ..utils.transforms import fliplr_joints, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CEP(data.Dataset):
    """
    """
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
            self.data_root = cfg.DATASET.TRAINROOT
        else:
            self.csv_file = cfg.DATASET.TESTSET
            self.data_root = cfg.DATASET.TESTROOT

        self.is_train = is_train
        self.transform = transform
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR

        #rot_factor 是什么 （旋转角度)
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #idx 为图片索引

        # image_path = os.path.join(self.data_root,
        #                           self.landmarks_frame.iloc[idx, 0])
        # print("img idx : %d"%idx)
        if not self.is_train:
            img_index = idx+300
        else:
            img_index = idx

        img_path = os.path.join(self.data_root,
                                "%03d.bmp"%(img_index +1))
        
        # 无scale
        # scale = self.landmarks_frame.iloc[idx, 1]
        # box_size = self.landmarks_frame.iloc[idx, 2]

        # center_w = self.landmarks_frame.iloc[idx, 3]
        # center_h = self.landmarks_frame.iloc[idx, 4]
        # center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 1:].values
        ##pts为所有Landmark的坐标点        shape (19,2)
        pts = pts.astype('float').reshape(-1, 2)


        # scale *= 1.25
        nparts = pts.shape[0] #Landmark的数量
        # img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        img = np.array(Image.open(img_path), dtype=np.float32)
        img = img[:, :, np.newaxis]

        # r = 0
        # if self.is_train:
            #scale 会影响什么部分？
            # scale = scale * (random.uniform(1 - self.scale_factor,
            #                                 1 + self.scale_factor))

            ##图像增强中的旋转和翻转
            # r = random.uniform(-self.rot_factor, self.rot_factor) \
            #     if random.random() <= 0.6 else 0
            # if random.random() <= 0.5 and self.flip:
            #     img = np.fliplr(img)
            #     pts = fliplr_joints(pts, width=img.shape[1], dataset='CEP')
                # center[0] = img.shape[1] - center[0]

        #图像根据上述增强进行变化
        # img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        ##target 为生成的HeatMap   shape:19 x 64 x 64

        # 自己添加的坐标变换，读入的明明是原图的坐标 但是generate heatmap的时候却用了64x64下的坐标
        # 是在transform_pixel的时候处理了
        tpts = pts.copy() * [float(self.output_size[0])/float(self.input_size[0]) ,
                             float(self.output_size[0])/float(self.input_size[0])]

        ##复制一份Landmark坐标点
        ##transformed points


        for i in range(nparts):
            #逐个坐标点遍历
            if tpts[i, 1] > 0:
                #如果y坐标>0 ?
                # 将Landmark进行对应的角度变化和 scale的变化 ?
                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                #                                scale, self.output_size, rot=r)

                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1,
                #                                self.output_size)

                # 生成heatmap图 target[i] 表示第i个点的Heatmap
                # sigma为超参数
                # 在这部分可能有精度的损失,每次产生的Heatmap不同
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)


        ##归一化与ToTensor
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        # img = img.squeeze(2)
        # print(img.shape)
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)

        tpts = torch.Tensor(tpts)   #转化后所有的的坐标值
        # center = torch.Tensor(center) #当前

        # meta = {'index': idx, 'center': center, 'scale': scale,
        #         'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta


if __name__ == '__main__':

    pass
