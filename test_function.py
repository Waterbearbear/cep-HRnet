# from PIL import Image, ImageFile
# import os
# import numpy as np
import torch
import lib.core.function as function
from lib.utils import transforms
import numpy as np
from matplotlib import pyplot as plt
from lib.core.evaluation import decode_preds,compute_nme
from torch.utils.data import DataLoader
from lib.config import config, update_config
from lib.datasets import get_dataset


#
# img_path = "E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\images\TrainDataCrop256"
# img_num = 1
#
# img_path = os.path.join(img_path,'%03d.bmp'%img_num)
#
#
#
# # img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
# img = np.array(Image.open(img_path),dtype=np.float32)
# img = img[:,:,np.newaxis]
#
# print(type(img))
# print(img.shape)
# print(img.dtype.name)


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
    # print(maxval.shape)
    # print(idx.shape)
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
    print(preds)

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # print(pred_mask)
    # print(pred_mask.shape)
    preds *= pred_mask
    print(preds)
    # pred_mask 的作用是什么？
    # preds.shape = [1,2,38]

    return preds


if __name__ == "__main__":

    gpus = list(config.GPUS)

    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)


    for i,(img,heatmap,meta) in enumerate(train_loader):
        if i == 1:
        #
        #     print(meta['pts'].shape)
        #     print(meta['pts'])
        #
        #
        #     landmark = 2
        #     batch = 0
        #     pred = decode_preds(heatmap,[64,64])
        #     print(pred.shape)
        #     print(pred)
        #
        #     nme_batch = compute_nme(pred, meta)
        #     print(nme_batch)
        #
        #     pred_x ,pred_y = pred[batch,landmark,:2].numpy()
        #     plt.imshow(heatmap[batch][landmark])
        #     # point_x,point_y = meta["tpts"][landmark,:2].numpy()
        #     # print(meta["tpts"].shape)
        #     point_x, point_y = meta["tpts"][batch,landmark, :2].numpy()
        #
        #
        #     plt.plot(point_x,point_y,".r")
        #     # plt.plot(pred_x,pred_y,".b")
        #     plt.title("batch: %d landmark: %d"%(batch,landmark))
        #     plt.grid(True)
        #     plt.colorbar()
        #     plt.show()
        #     break
            print("loop index : %d"%i)
            print(img.shape)
            img = img.numpy()
            print(img.shape)
            img = np.reshape(img, [2, 512, 512])
            print(img.shape)

            heatmap = torch.sum(heatmap,dim = 1)


            plt.figure()
            # plt.imshow(img[1], cmap=plt.cm.gray)
            plt.imshow(heatmap[1])
            plt.pause(0.001)
            plt.show()



            break




    # a = torch.rand([8,19,64,64])

    # heatmap = np.zeros((2,19,64,64))
    # landmark = [34,21]
    #
    # heatmap = transforms.generate_target(heatmap,landmark,1.5)
    #
    # pred = decode_preds(heatmap,[64,64])
    # print(heatmap)



    # coord = get_preds(a)

    # print(coord.shape)