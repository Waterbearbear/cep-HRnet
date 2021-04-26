import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import lib.core.function as function
from lib.utils import transforms
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from skimage import io




def nme_plot(preds ,meta):

    # targets 居然是256x256下的坐标
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    # L为landmark数量
    # (N -- batch_size , L --- 19)

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
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


def show_landmarks(data,idx, groundtruth,predtions):
    """Show image with landmarks"""
    #idx从0开始
    if data == "Test":
        img_idx =  idx + 301
    else:
        img_idx = idx

    img_path = os.path.join(".\data\%sDataCrop256" % data,
                            "%03d.bmp" % img_idx)


    image = io.imread(img_path)

    # landmarks_1 = groundtruth.iloc[idx - 1 - 300,1:].values


    plt.figure()
    plt.imshow(image,cmap=plt.cm.gray)


    plt.scatter(groundtruth[idx,:,0], groundtruth[idx,:, 1], s=10, marker='.', c='r')
    plt.scatter(predtions[idx,:, 0], predtions[idx,:, 1], s=10, marker='.', c='b')


    plt.title("%s %d prediction"%(data,img_idx))
    plt.savefig(".\output\CEP\cep_hrnet_w18\\CEP_predict%d.png"%img_idx)
    plt.pause(0.01)  # pause a bit so that plots are updated
    plt.show()


# def plotacc(size,y1,y2,y3,y4):
#
#
#
#
#     x = ["%d mm"%(i) for i in [2,2.5,3,4]]
#     total_width, n = 0.8, 4  # 有多少个类型，只需更改n即可
#     width = total_width / n
#     x = x - (total_width - width) / 2
#
#     plt.figure()
#     plt.bar(x, y1, width=width, label='2mm', color='darkorange')
#     # plt.bar(x + width, y2, width=width, label='2.5mm', color='deepskyblue')
#     # plt.bar(x + 2 * width, y3, width=width, label='3mm', color='green')
#     # plt.bar(x + 3 * width,y4,width = width ,label = '4mm', color = 'gold')
#
#     plt.xticks()
#     plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
#     # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.xlabel('landmarks')
#     plt.ylabel('accuracy')
#     # plt.rcParams['savefig.dpi'] = 300  # 图片像素
#     # plt.rcParams['figure.dpi'] = 300  # 分辨率
#     plt.rcParams['figure.figsize'] = (30.0, 16.0)  # 尺寸
#     plt.title("Accuracy ")
#     plt.savefig('E:\BME\HRNet-Facial-Landmark-Detection-master\output\CEP\cep_hrnet_w18\\acc.png')
#     plt.show()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 打印出19个点的MRE
    # 显示预测的点和ground truth的对比图
    # 显示准确率 在2mm  2.5mm 3mm 4mm

    #
    #  转换公式有问题 精度损失?
    #  试着打印256图的pred和ground truth对比

    path_test = ".\data\Landmark_Test.csv"
    path_test256 = ".\data\Landmark_TestCrop256.csv"
    path_test512 = ".\data\Landmark_TestCrop512.csv"

    path_pred = ".\output\CEP\cep_hrnet_w18\predictions.pth"

    pred_x_512,pred_y_512 = 512,512
    pred_x_256,pred_y_256 = 256,256
    crop_x,crop_y = 1678,1678
    crop_size_x = 142
    crop_size_y = 682


    transform = np.array([(pred_x_256/pred_x_256),( pred_y_256/pred_x_256 )])
    # print(transform)
    # print(transform.shape)

    #读取模型输出的结果 转换为1940x2400下的坐标
    #这一步确实会有精度损失 , 试着计算512x512图下的NME?
    # MRE很好，是测试集的问题


    pred_ori = torch.load(path_pred).numpy()
    pred = (pred_ori * transform)
    # pred = (pred_ori * transform) + np.array([crop_size_x,crop_size_y])

    #读取Ground Truth的坐标
    # 修改path_test512  获取不同分辨率下的坐标
    landmarks_frame = pd.read_csv(path_test256)


    landmarks  = landmarks_frame.iloc[:, 1:].values
    landmarks  = landmarks.astype('float').reshape(-1,19,2)

    rmse = np.zeros([19,1])

    rmse_acc = np.linalg.norm(pred - landmarks, axis=2)
    # rmse_acc2 =  np.sum((rmse_acc  <= 2))/pred.shape[0]  /19
    # rmse_acc25 = np.sum((rmse_acc <= 2.5))/pred.shape[0] /19
    # rmse_acc3  = np.sum((rmse_acc <= 3))/pred.shape[0]   /19
    # rmse_acc4  = np.sum((rmse_acc <= 4))/pred.shape[0]   /19


    # 不同scale下的准确率
    rmse_acc_list = []
    for i in [2,2.5,3,4]:
        rmse_ac = np.sum((rmse_acc) <= i)/pred.shape[0] /10
        rmse_acc_list.append(rmse_ac)

    names_acc = ["%0.1f mm"%i for i in [2,2.5,3,4]]
    plt.figure()
    plt.bar(names_acc, rmse_acc_list)
    plt.xlabel("scale")
    plt.ylabel("Predict Accuracy")
    # plt.title("")
    plt.savefig(".\output\CEP\cep_hrnet_w18\\acc.png")
    plt.show()



    #对19个landmarks计算rmse
    for i in range(rmse.shape[0]):
        rmse[i] = np.sum(np.linalg.norm(pred[:,i,:] - landmarks[:,i,:], axis=1))/pred.shape[0]


    #转换为毫米误差
    # rmse *= 0.1
    names = ["%d"%(i+1) for i in range(rmse.shape[0])]
    # print(names)
    rmse = np.reshape(rmse,[19])
    # print(rmse.shape)
    # 打印rmse

    plt.figure()
    plt.bar(names,rmse)
    plt.xlabel("landmarks")
    plt.ylabel("MRE(mm)")
    plt.title("MRE")
    plt.savefig(".\output\CEP\cep_hrnet_w18\\MRE.png")
    plt.show()

    # plotacc(19,rmse_acc2,rmse_acc25,rmse_acc3,rmse_acc4)
    # landmarks_frame = pd.read_csv(path_test256)
    # landmarks_frame = pd.read_csv(path_test512)
    # landmarks = landmarks_frame.iloc[:, 1:].values
    # landmarks = landmarks.astype('float').reshape(-1, 19, 2)
    # show_landmarks("Test",4,landmarks,pred_ori)

    # 原图中 模型输出与Ground Truth的对比
    show_landmarks("Test",0,landmarks,pred)

    # landmarks_gt = landmarks_frame.iloc[i - 1 - 300, 1:].values