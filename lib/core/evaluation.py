# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    ## score map是heatmap

    # scores = [batch_size,channels,img_size,img_size]
    # scores.dim() == 4
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    #将每一张heatmap展开成1维张量进行预测
    #maxval,idx      shape = ([batch_size,channels])


    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1
    # #maxval,idx      shape = ([batch_size,channels,1])

    preds = idx.repeat(1, 1, 2).float()
    # repeat 对相应维度复制几次
    # preds = ([batch_size,channels,2])
    # 用于生成x,y坐标

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1  #x坐标
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1 #y坐标

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    # pred_mask 的作用是什么？
    # preds.shape = [batch_size,channels,2]
    # 现在得到的preds 是 64x64图下的坐标
    # 我现在要那么将nme里的target转为64的 要么将这个preds转为256的


    return preds


def compute_nme(preds, meta):
    #  preds.shape [batch_size,channels,2]  每个channel下的预测点坐标
    #  meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}


    #targets 居然是256x256下的坐标
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()


    #L为landmark数量
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        # 按Batch进行遍历
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            # interocular = meta['box_size'][i]
            pass
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        # rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (L)

    return rmse


# def decode_preds(output, center, scale, res):
def decode_preds(output,  res):

    #output : 网络的总输出  这里为score_map  即Heatmap
    #center : AFLW的center坐标  区别于landmark
    #scale  : 未知
    #res    : heatmap的大小 这里为[64,64]

    coords = get_preds(output)  # float type
    # coords.shape = [batch_size,channels,2]

    coords = coords.cpu()

    # pose-processing
    # for n in range(coords.size(0)):
    #     #batch中遍历
    #     for p in range(coords.size(1)):
    #         #19个landmark遍历
    #         # heat map
    #         hm = output[n][p]
    #         # 获取Landmark的x,y坐标
    #         px = int(math.floor(coords[n][p][0]))
    #         py = int(math.floor(coords[n][p][1]))
    #         if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
    #             diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
    #             coords[n][p] += diff.sign() * .25
    #             #               -1  , if x < 0,
    #             #y = sign(x) =   0  , if x = 0,
    #             #                1  , if x > 0.
    #             #做这个变换是为了什么?  加了之后就飘了 迷
    #
    #
    # coords += 0.5
    # print(coords)
    # print(coords.shape)
    preds = coords.clone() * (256/64)


    # Transform back
    # for i in range(coords.size(0)):
    #     preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

