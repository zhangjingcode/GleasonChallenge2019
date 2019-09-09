import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import h5py

from Utility.ReadAndSave import ReadCoreImg, ReadLabelingImg
from Utility.ArrayProcess import OneHot
from Utility.Visulization import ShowOneHot,ShowH5
from MeDIT.SaveAndLoad import SaveH5

def GetAllCoreImgPath(core_img_folder):
    return [os.path.join(core_img_folder, img_index) for img_index in os.listdir(core_img_folder) if
            os.path.splitext(img_index)[-1] == '.jpg']

def GetAllAnnotationImgPath(annotation_img_folder, pathologists_num=6):
    all_annotation_path_list = []
    for pathologists_index in range(pathologists_num):
        sub_annotation_path = os.path.join(annotation_img_folder, 'Maps'+str(pathologists_index+1)+'_T')
        sub_annotation_path_list = [os.path.join(sub_annotation_path, img_index) for img_index in
                                    os.listdir(sub_annotation_path) if os.path.splitext(img_index)[-1] == '.png']
        all_annotation_path_list.append(sub_annotation_path_list)
    return all_annotation_path_list

def CheckSingleCase(core_img_path, all_annotation_folder_path_list):
    core_index = os.path.split(core_img_path)[-1].replace('.jpg', '')

    annotation_path_list = []
    for pathologists_index in range(len(all_annotation_folder_path_list)):
        for sub_annotation_img_path in all_annotation_folder_path_list[pathologists_index]:
            if sub_annotation_img_path.rfind(core_index) != -1:
                annotation_path_list.append(sub_annotation_img_path)

    # for index in annotation_path_list:
    #     print(index)
    return annotation_path_list

def MergeAnnotation(core_img_path, annotation_img_path_list, store_path='', show=False):
    # TODO: 1. 固定横纵比，2. 先One-hot编码，再进行Voting
    core_img_array, case_name = ReadCoreImg(core_img_path)

    col = core_img_array.shape[1]
    row = core_img_array.shape[0]


    core_img_array = cv2.resize(core_img_array, (col//10, row//10), interpolation=cv2.INTER_CUBIC)
    annotation_dict = {}

    merged_annotation_array = np.zeros((core_img_array.shape[0], core_img_array.shape[1], 6, len(annotation_img_path_list)))
    for annotation_index in range(len(annotation_img_path_list)):
        annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path_list[annotation_index])
        annotation_img_array = cv2.resize(cv2.resize(annotation_img_array, (col//10, row//10), interpolation=cv2.INTER_NEAREST),
                                          (col//10, row//10), interpolation=cv2.INTER_NEAREST)

        #median filter
        annotation_img_array = cv2.medianBlur(annotation_img_array, 9)

        annotation_dict[pathologist_num] = OneHot(annotation_img_array)
        if show:
            ShowOneHot(core_img_array, OneHot(annotation_img_array), title=case_name+'_'+pathologist_num, store_path=
                   os.path.join(store_path, case_name+'_'+pathologist_num+'.jpg'))

        merged_annotation_array[..., annotation_index] = OneHot(annotation_img_array)

    # merged_annotation_one_hot_array = OneHot(merged_annotation_array)
    modal_merged_annotation_array = np.zeros((merged_annotation_array.shape[:3]))
    for row in range(merged_annotation_array.shape[0]):
        for col in range(merged_annotation_array.shape[1]):
            print(row, col)
            annotation_array = np.array(merged_annotation_array[row, col, ...], dtype=np.int64)

            modal_merged_annotation_array[row, col, :] = np.sum(annotation_array, axis=1)/annotation_array.shape[-1]

    if show:
        ShowOneHot(core_img_array, modal_merged_annotation_array, title=case_name, store_path=
                   os.path.join(store_path, case_name+'.jpg'))

    if store_path:
        h5_store_path = os.path.join(store_path, case_name+'.h5')
        SaveH5(h5_store_path,
               [core_img_array / 128 - 1, modal_merged_annotation_array],
               tag=['input_0', 'output_0'],
               data_type=[np.float, np.float])

        # with h5py.File(h5_store_path, 'w') as h5_file:
        #     h5_file['input_0'] = core_img_array / 128 - 1
        #     h5_file['output_0'] = OneHot(modal_merged_annotation_array)

    return modal_merged_annotation_array



def GenerationH5(core_img_folder, store_folder):
    for sub_file in os.listdir(core_img_folder):
        if sub_file.rfind('.jpg') != -1:
            sub_file_path = os.path.join(core_img_folder, sub_file)
            print(sub_file_path)

            MergeAnnotation(sub_file_path,
                            CheckSingleCase(sub_file_path, GetAllAnnotationImgPath(r'W:\MRIData\OpenData\Gleason2019')),
                            store_path=store_folder, show=False)

def CheckOneMergedH5():
    from CustomerPath import one_h5_predict_path
    from MeDIT.SaveAndLoad import LoadH5
    # with open(one_h5_predict_path, 'rb') as file:
    #     f = h5py.File(file)
    #     print(f.keys())
    #
    # data = LoadH5(one_h5_merged_file_path, tag='input_0', read_model='r')
    label = LoadH5(one_h5_predict_path, tag='output_0', read_model='r')
    predict = LoadH5(one_h5_predict_path, tag='predict_0', read_model='r')

    show_label = np.argmax(label, axis=-1).astype(float)
    show_predict = np.argmax(predict, axis=-1).astype(float)

    from scipy import signal
    kernel_size = (25, 25)
    filted_label = signal.medfilt2d(show_label, kernel_size=kernel_size)
    filted_predict = signal.medfilt2d(show_predict, kernel_size=kernel_size)

    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(show_label, vmax=5, vmin=0, cmap='jet')
    plt.title('Merged Label')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(filted_label, vmax=5, vmin=0, cmap='jet')
    plt.title('Fitered Label {}'.format(kernel_size))
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(show_predict, vmax=5, vmin=0, cmap='jet')
    plt.title('Merged predict')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(filted_predict, vmax=5, vmin=0, cmap='jet')
    plt.title('Fitered predict {}'.format(kernel_size))
    plt.colorbar()
    plt.show()



