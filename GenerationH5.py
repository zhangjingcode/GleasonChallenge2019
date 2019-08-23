import h5py
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from CustomerPath import core_img_path, annotation_img_path, store_path
from ImageDraw import ReadCoreImg, ReadLabelingImg


def SingleCaseGeneration(core_img_path, annotation_img_path, store_path):

    core_img_array, case_name = ReadCoreImg(core_img_path)
    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)

    downsampled_core_img_array = cv2.resize(core_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)
    downsampled_annotation_img_array = cv2.resize(annotation_img_array, (512, 512), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(core_img_array)
    plt.show()

    h5_file_path = os.path.join(store_path, case_name+'_'+pathologist_num+'.h5')
    with h5py.File(h5_file_path, 'w') as target_h5_file:
        target_h5_file['input_0'] = downsampled_core_img_array
        target_h5_file['output_0'] = downsampled_annotation_img_array

def CheckH5(h5_file_path):

    plt.subplot(121)
    with h5py.File(h5_file_path, 'r') as target_h5_file:
        downsampled_core_img_array = target_h5_file['input_0'].value
        downsampled_annotation_img_array = target_h5_file['output_0'].value

    plt.imshow(downsampled_core_img_array)
    # plt.contour(downsampled_annotation_img_array)
    plt.title('input_0' +'\n'+str(downsampled_core_img_array.shape))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(downsampled_annotation_img_array*255)
    plt.title('output_0' + '\n'+str(downsampled_annotation_img_array.shape))


    plt.xticks([])
    plt.yticks([])
    plt.show()

# SingleCaseGeneration(core_img_path, annotation_img_path, store_path)

CheckH5()
