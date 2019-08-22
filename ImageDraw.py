import cv2
import os
import re
import numpy as np

from collections import Counter
from skimage import measure,draw

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



def MarginPlot(img_path, label_path,case_name,label_num):
    img_array = mpimg.imread(img_path)
    label_array = mpimg.imread(label_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    #show img
    ax.imshow(img_array, plt.cm.gray)

    #show margin
    print(Counter(label_array.flatten()))
    ##此方法无法识别中空部位，会视为另一种区域
    contours = measure.find_contours(label_array, 0.0117)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=6)

    ax.set_title(case_name+' in Maps '+label_num)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return ax

# MarginPlot(r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide002_core033.jpg',
#            r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide002_core033_classimg_nonconvex.png',
#            'slide001_core037', '1')

def Iteration(train_folder, label_folder):
    for sub_file in os.listdir(train_folder):
        case_name = sub_file.replace('.jpg', '')
        case_path = os.path.join(train_folder, case_name+'.jpg')

        label_path = os.path.join(label_folder, case_name+'_classimg_nonconvex.png')

    #get the label num
        label_num = re.findall('[1-9]', os.path.split(label_folder)[-1])


        if not os.path.exists(label_path):
            print(case_name+' is not exist in label !')
        else:
            img = MarginPlot(case_path, label_path, case_name, label_num[0])


# Iteration(r'W:\MRIData\OpenData\Gleason2019\Train Imgs',
#           r'W:\MRIData\OpenData\Gleason2019\Maps1_T')

def ReadSingleImgByMatplot():
    # img_path = r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide001_core003.jpg'
    label_path = r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide002_core033_classimg_nonconvex.png'
    #
    # img = mpimg.imread(img_path)
    label = mpimg.imread(label_path)

    contours = measure.find_contours(label, 0.5)

    max = np.max(label)

    # MarginPlot(img, label)
    counter = Counter(label.flatten())
    print(counter)
    # print(Counter(img.flatten()))
    # img = MarginPlot(label)
    plt.imshow(label)
    plt.axis('off')
    plt.show()

# ReadSingleImgByMatplot()

def ReadImgbyCV2():
    import cv2  # 导入模块，opencv的python模块叫cv2
    # imgobj = cv2.imread(r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core037_classimg_nonconvex.png')
    # imgobj = cv2.imread(r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core008_classimg_nonconvex.png', 1)
    imgobj = cv2.imread(r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide002_core033_classimg_nonconvex.png',1)
    # 读取图像
    # cv2.namedWindow("image")  # 创建窗口并显示的是图像类型
    # print(np.max(imgobj))
    # plt.imshow(imgobj)
    # plt.show()
    print(Counter(imgobj.flatten()))
    imgobj=imgobj*255
    print(np.max(imgobj))
    plt.imshow(imgobj,cmap='winter')
    plt.show()
    # cv2.imshow("image", imgobj)
    # cv2.waitKey(0)  # 等待事件触发，参数0表示永久等待
    # cv2.destroyAllWindows()
#
ReadImgbyCV2()