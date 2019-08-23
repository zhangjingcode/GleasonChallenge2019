import cv2
import os
import re
import numpy as np
from CustomerPath import core_img_path, annotation_img_path
from collections import Counter
from skimage import measure,draw

import matplotlib.image as mpimg
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

    # read case information
    pathologist_num = annotation_img_path.split('\\')[4]

    return annotation_img_array[:, :, 0], pathologist_num

def SeparateLabel(annotation_img_array):

    seperate_label_array_dict = {}

    # Separate 1, 3, 4, 5, 6 voxel array
    print(Counter(annotation_img_array.flatten()))
    for voxel_threshold in [1, 3, 4, 5, 6]:
        if voxel_threshold in annotation_img_array:
            sub_annotation_img_array = np.zeros(annotation_img_array.shape)
            sub_annotation_img_array[np.where(annotation_img_array == voxel_threshold)] = voxel_threshold
            seperate_label_array_dict[voxel_threshold] = sub_annotation_img_array
    return seperate_label_array_dict

def OriginalImgPlot(core_img_path, annotation_img_path):
    '''
    Draw the margin of annotation image in core imag path
    :param core_img_path: image file path of core pathology image
    :param annotation_img_path: image file path of pathologist annotation img
    :return: ax object
    '''

    # Read Img and info

    core_img_array, case_name = ReadCoreImg(core_img_path)
    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)
    title = case_name+' in Maps of '+pathologist_num

    # Show margin
    ax = MarginPlot(core_img_array, annotation_img_array, title+' '+str(core_img_array.shape))
    return ax


def DownSamplingImgPlot(core_img_path, annotation_img_path):
    '''
    Draw the margin of annotation image in core imag path
    :param core_img_path: image file path of core pathology image
    :param annotation_img_path: image file path of pathologist annotation img
    :return: ax object
    '''

    # Read Img and info

    core_img_array, case_name = ReadCoreImg(core_img_path)
    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)
    title = case_name+' in Maps of '+pathologist_num

    #resize to (512,512)
    downsampled_core_img_array = cv2.resize(core_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)
    downsampled_annotation_img_array = cv2.resize(annotation_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)

    # downsampled_core_img_array = cv2.pyrDown(core_img_array)
    # downsampled_annotation_img_array = cv2.pyrDown(annotation_img_array)
    # print('shape after downsampling ', downsampled_core_img_array.shape)

    # Show margin
    ax = MarginPlot(downsampled_core_img_array, downsampled_annotation_img_array,
               title+' '+str(downsampled_core_img_array.shape))

    return ax


def MarginPlot(core_img_array, annotation_img_array, title):
    """

    :param core_img_array: the array of img want to show
    :param annotation_img_array: the array of annotation want to show
    :param title:
    :return: ax object
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # show img

    ax.imshow(core_img_array)

    # set color

    color_map = {1:'lawngreen', 3:'darkorange', 4: 'blue', 5: 'red', 6: 'black'}

    # separate label array

    separate_label_array_dict = SeparateLabel(annotation_img_array[:, :, 0])

    # show contour in different voxel threshold

    for voxel_threshold in [1, 3, 4, 5, 6]:
        if voxel_threshold in separate_label_array_dict.keys():
            seperate_label_array = separate_label_array_dict[voxel_threshold]
            ax.contour(seperate_label_array, colors=color_map[voxel_threshold], linewidths=1, linestyles='dotted')
            # add special label for margin
            ax.plot(0, 0, '-', label=voxel_threshold, color=color_map[voxel_threshold])

    # cs = ax.contour(annotation_img_array[:,:,0], levels=[1, 3, 4, 5, 6],
    #             cmap="Accent", linewidths=5)
    # fig.colorbar(cs, ax=ax,extendfrac=False)

    ax.legend()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()
    return ax

def MergeOrigianlImgAndDownSampledImg(core_img_path, annotation_img_path):


    # Read Img and info

    core_img_array, case_name = ReadCoreImg(core_img_path)
    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)
    title = case_name+' in Maps of '+pathologist_num

    #resize to (512,512)
    downsampled_core_img_array = cv2.resize(core_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)
    downsampled_annotation_img_array = cv2.resize(annotation_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)

    fig = plt.figure()
    fig.suptitle(title)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.imshow(core_img_array)
    ax1.set_title(str(core_img_array.shape))
    ax1.set_xticks([])
    ax1.set_yticks([])


    # set color

    color_map = {1:'lawngreen', 3:'darkorange', 4: 'blue', 5: 'red', 6: 'black'}

    # separate label array

    separate_label_array_dict = SeparateLabel(annotation_img_array)

    for voxel_threshold in [1, 3, 4, 5, 6]:
        if voxel_threshold in separate_label_array_dict.keys():
            seperate_label_array = separate_label_array_dict[voxel_threshold]
            ax1.contour(seperate_label_array, colors=color_map[voxel_threshold], linewidths=2, linestyles='dotted')
            # add special label for margin
            ax1.plot(0, 0, '-', label=voxel_threshold, color=color_map[voxel_threshold])

    ax1.legend()

    ax2.imshow(downsampled_core_img_array)
    ax2.set_title(str(downsampled_core_img_array.shape)+' with INTER_NEAREST')
    ax2.set_xticks([])
    ax2.set_yticks([])


    separate_label_array_dict = SeparateLabel(downsampled_annotation_img_array)

    for voxel_threshold in [1, 3, 4, 5, 6]:
        if voxel_threshold in separate_label_array_dict.keys():
            seperate_label_array = separate_label_array_dict[voxel_threshold]
            ax2.contour(seperate_label_array, colors=color_map[voxel_threshold], linewidths=1, linestyles='dotted')
            # add special label for margin
            ax2.plot(0, 0, '-', label=voxel_threshold, color=color_map[voxel_threshold])
    ax2.legend()
    plt.show()

# core_img_path = r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide006_core110.jpg'
# annotation_path = r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide006_core110_classimg_nonconvex.png'
# DownSamplingImgPlot(core_img_path, annotation_img_path)
# MergeOrigianlImgAndDownSampledImg(core_img_path, annotation_img_path)


