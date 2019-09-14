import h5py
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# from CustomerPath import core_img_path, annotation_img_path, store_path
from Utility.ReadAndSave import ReadLabelingImg, ReadCoreImg
from Utility.Visulization import ShowH5
from Utility.ArrayProcess import OneHot



def SingleCaseGeneration(core_img_path, annotation_img_path, store_path):

    core_img_array, case_name = ReadCoreImg(core_img_path)

    annotation_img_array, pathologist_num = ReadLabelingImg(annotation_img_path)

    col = core_img_array.shape[1]
    row = core_img_array.shape[0]

    downsampled_core_img_array = cv2.resize(core_img_array, (col//10, row//10), interpolation=cv2.INTER_NEAREST)
    downsampled_annotation_img_array = cv2.resize(annotation_img_array, (col//10, row//10), interpolation=cv2.INTER_NEAREST)

    with h5py.File(store_path, 'w') as target_h5_file:
        target_h5_file['input_0'] = downsampled_core_img_array / 255
        target_h5_file['output_0'] = OneHot(downsampled_annotation_img_array)




# SingleCaseGeneration(core_img_path, annotation_img_path, store_path)
def Iteration(core_img_foler, annotation_img_foler, store_folder):
    for sub_file in os.listdir(core_img_foler):

        core_img_name = sub_file.replace('.jpg', '')
        print(core_img_name)
        core_img_path = os.path.join(core_img_foler, core_img_name + '.jpg')

        annotation_img_path = os.path.join(annotation_img_foler, core_img_name + '_classimg_nonconvex.png')

        # get the label num
        annotation_img_num = os.path.split(annotation_img_foler)[-1]

        store_path = os.path.join(store_folder, core_img_name+'_'+annotation_img_num+'.h5')
        if not os.path.exists(annotation_img_path):
            print(core_img_name + ' is not exist in label !')
        else:
            SingleCaseGeneration(core_img_path, annotation_img_path, store_path)

def ItrationCheck(folder,store_path):
    for sub_file in os.listdir(folder):
        if os.path.split(sub_file)[-1].rfind('.h5') != -1:
            sub_file_path = os.path.join(folder, sub_file)
            sub_store_path = os.path.join(store_path, sub_file.replace('.h5', '.png'))

            ShowH5(sub_file_path, store_path=sub_store_path)

def main():
    # from CustomerPath import one_h5_check_store_path, one_h5_file_path
    one_h5_file_path =r'D:\Gleason2019\Train Imgs\slide001_core011.jpg'
    one_h5_check_store_path = r'C:\Users\zj\Desktop\SHGH\demo.h5'
    SingleCaseGeneration(one_h5_file_path, r'D:\Gleason2019\Maps1_T\slide001_core011_classimg_nonconvex.png', one_h5_check_store_path)
    # ShowH5(one_h5_check_store_path, show=True)

    # Iteration(r'Y:\MRIData\OpenData\Gleason2019\Train Imgs',
    #           r'Y:\MRIData\OpenData\Gleason2019\Maps3_T',
    #           # r'X:\PrcoessedData\Challenge_Gleason2019\ProcessedH5_Maps3_256')

if __name__ == '__main__':
     main()



