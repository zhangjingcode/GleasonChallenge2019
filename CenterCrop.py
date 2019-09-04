import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from ImageDraw import ReadLabelingImg, ReadCoreImg

def CenterCrop(img_array, img_size):
    row = img_array.shape[0]
    col = img_array.shape[1]
    center_position = (row//2, col//2)

    distance = img_size//2
    croped_array = img_array[center_position[0]-distance:center_position[0]+distance,
                 center_position[1] - distance:center_position[1] + distance, :]

    print('Original img shape :{}  \n Croped img shape :{}'.format(img_array.shape, croped_array.shape))

    fig = plt.figure()
    fig.suptitle('CenterCrop')

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.imshow(img_array)
    ax1.set_title(str(img_array.shape))
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(croped_array)
    ax2.set_title(str(croped_array.shape))
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()
    return croped_array


img_array , case_name = ReadCoreImg(r'D:\Gleason2019\Train Imgs\slide001_core003.jpg')
CenterCrop(img_array, 4608)