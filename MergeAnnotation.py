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
# CheckSingleCase('slide001_core003', GetAllAnnotationImgPath(r'W:\MRIData\OpenData\Gleason2019'))
# core_img_path = r'W:\MRIData\OpenData\Gleason2019\Train Imgs\slide002_core041.jpg'
# MergeAnnotation(core_img_path,
#                 CheckSingleCase(core_img_path, GetAllAnnotationImgPath(r'W:\MRIData\OpenData\Gleason2019')), show=True,
#                 store_path=r'C:\Users\zj\Desktop\word_and_ppt\GleasonChallenge\OneHotVote')

GenerationH5(r'W:\MRIData\OpenData\Gleason2019\Train Imgs',
             r'V:\PrcoessedData\Challenge_Gleason2019\ProcessedH5_voted_10down\all_data')
