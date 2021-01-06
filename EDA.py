##获取landmark 坐标 并且将其存入CSV中
# 具体文件为landmark_Train.csv Landmark_Test.csv...


import os
import numpy as np
import pandas as pd
from skimage import io, transform
# import numpy as np
import matplotlib.pyplot as plt

def getTargetCoord(annotationFile):
    voxelCoords = []
    with open(annotationFile) as f:
        tag = False
        for lines in f:
            if tag:
                curLine = lines.strip().split('\t')
                voxelCoords.append([int(curLine[-2]), int(curLine[-1])])
                #加入坐标
                #例如annotator301  734   945
                #加入[734,945]

                tag = False
            else:
                tag = True

    return np.array(voxelCoords)

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image,cmap=plt.cm.gray)
    plt.scatter(landmarks[:,0], landmarks[:, 1], s=50, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.savefig('1-landmark.png')


if __name__ == "__main__":

    state = "Produce"  #Produce or any
    data = "Test"      #Train or Test
    crop = True        #True or False
    h ,w = 1678 , 1678
    outputshape = 512
    new_h,new_w = int(outputshape) , int(outputshape)
    ############ 生成部分############################
    if state == "Produce":
        if data == "Train":
    # landmark_path = "E:\BME\DCGAN-pytorch\TestDataCrop\landmark_001.txt"
            landmarkOri_path = "E:\Hughie\GAN\Data\TrainLabel\landmark_001.txt"
            len = 301
        else:
            landmarkOri_path = "E:\Hughie\GAN\Data\TestLabel\landmark_001.txt"
            len = 101
    #
        landmark = getTargetCoord(landmarkOri_path)

        # print(landmark)
        # print(landmark[0])

        # landmark = np.reshape(landmark,(2,-1))
        # print(landmark[:,0])
        # print(landmark)

        Keypoint1_x =  (landmark[:,0] - 142)  * (new_w/w)
        Keypoint1_y =  (landmark[:,1] - 682) * (new_h/h)

        Key = pd.DataFrame({"Keypoint1_x": Keypoint1_x,
                            "Keypoint1_y": Keypoint1_y},index = [i for i in range(1,len)])


        for i in range(2,20):
            # landmark_path = "E:\BME\DCGAN-pytorch\TestDataCrop\landmark_%03d.txt"%i
            landmarkOri_path = "E:\Hughie\GAN\Data\%sLabel\landmark_%03d.txt"%(data,i)
            k_x = "Keypoint%d_x"%i
            k_y = "Keypoint%d_y"%i

            landmark = getTargetCoord(landmarkOri_path)
            Keypoint1_x = (landmark[:, 0]  - 142)  * (new_w / w)
            Keypoint1_y = (landmark[:, 1]  - 682)  * (new_h / h)

            Key[k_x] = Keypoint1_x
            Key[k_y] = Keypoint1_y


        print(Key)
        print(Key.shape)
        print(Key.head())
        # max_Key = Key.max()
        # min_Key = Key.min()
        # max_x = 0
        # max_y = 0
        # min_x = 10000
        # min_y = 10000
        # for i in range(1,20):
        #     k_x = "Keypoint%d_x" % i
        #     k_y = "Keypoint%d_y" % i
        #
        #
        #     if max_Key[k_x] > max_x:
        #         max_x = max_Key[k_x]
        #     if max_Key[k_y] > max_y:
        #         max_y = max_Key[k_y]
        #     if min_Key[k_x] < min_x:
        #         min_x = min_Key[k_x]
        #     if min_Key[k_y] < min_y:
        #         min_y = min_Key[k_y]
        #
        #
        # print("左上:(%d,%d)"%(min_x,min_y))
        # print("左下:(%d,%d)"%(min_x,max_y))
        # print("右上:(%d,%d)"%(max_x,min_y))
        # print("右下:(%d,%d)"%(max_x,max_y))
        # print("中间点:(%d,%d)"%((max_x + min_x)/2,(min_y + max_y)/2))
        # print("图像大小:%d x %d"%(max_x - min_x,max_y - min_y))

        Key.to_csv('E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\Landmark_%sCrop512.csv'%data, index=True)

##########################################################################

    # print(Key.loc[[1]]["Keypoint1_x"])
    else:
        if crop:
            Key = pd.read_csv("E:\BME\DCGAN-pytorch\Landmark_%sCrop.csv"%data)
        else:
            Key = pd.read_csv("E:\BME\DCGAN-pytorch\Landmark_%s.csv"%data)

        list_x = []
        list_y = []
        img_num = 69

        for i in range(1,20):
            k_x = "Keypoint%d_x" % i
            k_y = "Keypoint%d_y" % i

            list_x.append(Key[k_x][img_num-1])
            list_y.append(Key[k_y][img_num-1])


        print(list_x)
        print(list_y)
        array_xy = np.array(list(zip(list_x,list_y)))
        if crop:
            img = io.imread(os.path.join('E:\BME\DCGAN-pytorch\%sDataCrop'%data, "%03d.bmp"%(img_num+300)))
        else:
            img = io.imread(os.path.join('E:\Hughie\GAN\Data\%sData' % data, "%03d.bmp" % (img_num+300)))
        plt.figure()
        plt.title("%03d.bmp"%(img_num+300))
        show_landmarks(img,array_xy)

        plt.show()

        # print(zip(list_x,list_y))






