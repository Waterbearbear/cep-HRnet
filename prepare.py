# import cv2
import os
# import cepset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from EDA import show_landmarks
from skimage import io, transform ,img_as_ubyte
import pandas as pd


for i in range(1,301):
    data = "Train"
    new_h,new_w = 512 ,512

    img_path = os.path.join("E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\images\%sDataCrop"%data,"%03d.bmp"%i)
    # img = cv2.imread(img_path)
    image = io.imread(img_path)

    image = np.reshape(image[:, :, 0], (1678, 1678, 1))
    # print(image.dtype.name)
    print("%03d.bmp"%i)

    # if i ==1:
    #     print(img[0][0])
    # cropped = img[681:2359, 141:1819]
    resize_img = img_as_ubyte(image)
    # print(resize_img.dtype.name)
    resize_img = transform.resize(resize_img, (new_h, new_w))

    if i==1:
        print("img.shape: ")
        print(image.shape)
        print("resize shape:")
        print(resize_img.shape)
        print(type(image))
        print(type(resize_img))

    io.imsave("E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\images\%sDataCrop%d\%03d.bmp"%(data,new_h,i),resize_img)

    if data == "Train":
        index = i
    else:
        index = i - 300
    if index == 89 or index == 1 or index == 53:
        root = os.path.join("E:\BME\HRNet-Facial-Landmark-Detection-master\CEP", "Landmark_%sCrop%d.csv" % (data , new_h))

        landmarks_frame = pd.read_csv(root)

        landmarks = landmarks_frame.iloc[i-1-300, 1:].values

        landmarks = landmarks.astype('float').reshape(-1, 2)
        print(landmarks.shape)

        plt.figure()

        resize_img = np.reshape(resize_img,(new_h,new_w))


        show_landmarks(resize_img, landmarks)
        fig = plt.gcf()
        fig.savefig('test.png',dpi = 100)


        plt.show()





# Cep_dataset = cepset.CephaloDataset(csv_file='E:\BME\DCGAN-pytorch\Landmark_TrainCrop.csv',
#                                     root_dir='E:\BME\DCGAN-pytorch\TrainDataCrop',
#                                     transform=transforms.Compose([
#                                         cepset.Rescale(256),
#                                         cepset.ToTensor(),
#                                         cepset.Normalize(mean=  (0.5), std= (0.5),inplace = True)
#                                     ]))

#
# fig = plt.figure()
#
#



#
# for i in range(200):
#     data = "Train"
#
#
#     # sample = Cep_dataset[i]
#     #
#     # img , landmarks = sample["image"],sample["landmarks"]
#     # print(i, sample['image'].shape, sample['landmarks'].shape)
#     # img = img.view(256,256).numpy()
#     # print(img[0][0])
#     # print(img[124][123])
#
#
#
#
#
#     plt.figure()
#
#     show_landmarks(img,landmarks)
#
#
#     if i == 0:
#         plt.show()
#         break