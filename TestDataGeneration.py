import h5py
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from CustomerPath import test_core_img_folder, test_h5_store_folder
from Utility.ReadAndSave import ReadLabelingImg, ReadCoreImg
from Utility.Visulization import ShowH5
from Utility.ArrayProcess import OneHot


def SingleCaseGeneration(core_img_path, store_path):

    core_img_array, case_name = ReadCoreImg(core_img_path)


    # col = core_img_array.shape[1]
    # row = core_img_array.shape[0]
    col = 5120
    row = 5120

    downsampled_core_img_array = cv2.resize(core_img_array, (col//10, row//10), interpolation=cv2.INTER_NEAREST)

    with h5py.File(store_path, 'w') as target_h5_file:
        target_h5_file['input_0'] = downsampled_core_img_array / 255
        target_h5_file['original_size'] = [col, row]







def Iteration(core_img_foler,store_folder):
    for sub_file in os.listdir(core_img_foler):

        core_img_name = sub_file.replace('.jpg', '')
        print(core_img_name)
        core_img_path = os.path.join(core_img_foler, core_img_name + '.jpg')

        store_path = os.path.join(store_folder, core_img_name+'_'+'.h5')

        SingleCaseGeneration(core_img_path, store_path)

def CheckTestH5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as target_h5_file:
        keys_index = list(target_h5_file.keys())
        core_img_array = target_h5_file['input_0'].value
        original_size = target_h5_file['original_size'].value

    plt.title(original_size)
    plt.imshow(core_img_array)
    plt.axis('off')
    plt.show()

def ItrationCheck(folder,store_path):
    for sub_file in os.listdir(folder):
        if os.path.split(sub_file)[-1].rfind('.h5') != -1:
            sub_file_path = os.path.join(folder, sub_file)
            sub_store_path = os.path.join(store_path, sub_file.replace('.h5', '.png'))

            ShowH5(sub_file_path, store_path=sub_store_path)

def main():
    # from CustomerPath import one_h5_check_store_path, one_h5_file_path
    # CheckTestH5(r'D:\Gleason2019\Test_h5\slide001_core015_.h5')
    # SingleCaseGeneration(core_img_path, h5_store_path)
    Iteration(test_core_img_folder, test_h5_store_folder)


if __name__ == '__main__':

     main()