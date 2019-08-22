import cv2
import os
import re
import numpy as np

import matplotlib.pyplot as plt


def ReadCoreImg(core_img_path):
    """
    Read pathology img by cv2

    :param core_img_path: image file path of core pathology image
    :return: img_array and case name
    """

    # Read image
    core_img_array = cv2.imread(core_img_path)

    # read case infomation
    case_path = os.path.split(core_img_path)[-1]
    case_name = case_path.replace('.jpg', '')

    return core_img_array, case_name


def ReadLabelingImg(annotation_img_path):
    """
    Read pathologist annotation img by cv2
    :param labling_img_path: image file path of pathologist annotation img
    :return: labeling_array, pathologist' number
    """
    annotation_img_array = cv2.imread(annotation_img_path)

    # read case infomation
    pathologist_num = annotation_img_path.split('\\')[4]

    return annotation_img_array, pathologist_num



def MarginPlot(core_img_path, annotation_img_path):
    '''
    Draw the margin of annotation image in core imag path
    :param core_img_path: image file path of core pathology image
    :param annotation_img_path: image file path of pathologist annotation img
    :return: ax
    '''
    core_img_array, case_name = ReadCoreImg(core_img_path)
    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    #show img
    ax.imshow(core_img_array)

    ## draw the contour of annotation image
    cs = ax.contour(annotation_img_array[:,:,0], levels=[1, 3, 4, 5, 6],
                cmap="Accent", linewidths=3)
    fig.colorbar(cs, ax=ax,extendfrac=False)


    ax.set_title(case_name+' by '+pathologist_num)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return ax

MarginPlot(r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide002_core033.jpg',
           r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide002_core033_classimg_nonconvex.png')

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




