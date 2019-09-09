import os
import re

import cv2
import numpy as np
from collections import Counter

from skimage import measure, draw

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from .Utility.ReadAndSave import ReadCoreImg, ReadLabelingImg
# from CustomerPath import core_img_path, annotation_img_path





def MarginPlot(core_img_array, annotation_img_array, title):
    """

    :param core_img_array: the array of img want to show
    :param annotation_img_array: the array of annotation want to show
    :param title:
    :return: ax object
    """
    color_map = {1: 'lawngreen', 3: 'darkorange', 4: 'blue', 5: 'red', 6: 'black'}
    separate_label_array_dict = SeparateLabel(annotation_img_array)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(core_img_array)

    for specific_level in [1, 3, 4, 5, 6]:
        if specific_level in separate_label_array_dict.keys():
            seperate_label_array = separate_label_array_dict[specific_level]
            ax.contour(seperate_label_array, colors=color_map[specific_level], linewidths=1, linestyles='dotted')

            # add special label for margin
            ax.plot(0, 0, '-', label=specific_level, color=color_map[specific_level])

    ax.legend()
    ax.set_title(title)
    ax.axis('off')              # Double-check
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    return ax

############################################################

def OriginalImagePlot(core_img_path, annotation_img_path):

    """

    Draw the margin of annotation image in core imag path
    :param core_img_path: image file path of core pathology image
    :param annotation_img_path: image file path of pathologist annotation img
    :return: ax object
    """

    # Read Img and info
    core_img_array, case_name = ReadCoreImg(core_img_path)

    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)
    title = case_name + ' in Maps of ' + pathologist_num


    # Show margin
    ax = MarginPlot(core_img_array, annotation_img_array, title + ' ' + str(core_img_array.shape))
    return ax

def DownSamplingImgPlot(core_img_path, annotation_img_path):
    """

    Draw the margin of annotation image in core imag path
    :param core_img_path: image file path of core pathology image
    :param annotation_img_path: image file path of pathologist annotation img
    :return: ax object
    """

    # Read Img and info

    core_img_array, case_name = ReadCoreImg(core_img_path)

    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)

    row, column = core_img_array.shape[:2]



    # down-sampling 10 times
    downsampled_core_img_array = cv2.resize(core_img_array, (row // 10, column // 10), interpolation=cv2.INTER_CUBIC)
    downsampled_annotation_img_array = cv2.resize(annotation_img_array, (row // 10, column // 10), interpolation=cv2.INTER_NEAREST)

    title = '{} in Maps of {} {}'.format(case_name, pathologist_num, str(downsampled_core_img_array.shape))
    ax = MarginPlot(downsampled_core_img_array, downsampled_annotation_img_array, title)

    return ax


############################################################


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

    color_map = {1: 'lawngreen', 3: 'darkorange', 4: 'blue', 5: 'red', 6: 'black'}

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

    color_map = {1: 'lawngreen', 3: 'darkorange', 4: 'blue', 5: 'red', 6: 'black'}

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


if __name__ == '__main__':
    core_img_path = r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide006_core110.jpg'
    annotation_path = r'W:\MRIData\OpenData\Gleason2019\Maps1_T\slide006_core110_classimg_nonconvex.png'
    DownSamplingImgPlot(core_img_path, annotation_img_path)
    MergeOrigianlImgAndDownSampledImg(core_img_path, annotation_img_path)



