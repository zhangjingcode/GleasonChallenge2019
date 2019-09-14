import os
import re

import cv2
import numpy as np
import h5py
from collections import Counter

from skimage import measure, draw

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


def ReadLabelingImg(annotation_img_path, annotation_index):

    """
    Read pathologist annotation img by cv2
    :param Annotation_img_path: image file path of pathologist annotation img
    :return: labeling_array, pathologist' number
    """
    annotation_img_array = cv2.imread(annotation_img_path)

    # read case information
    pathologist_num = annotation_img_path.split('\\')[annotation_index]

    return annotation_img_array[:, :, 0], pathologist_num

