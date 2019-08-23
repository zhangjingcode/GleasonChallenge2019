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
    # plt.show()


    with h5py.File(store_path, 'w') as target_h5_file:
        target_h5_file['input_0'] = downsampled_core_img_array / 255
        target_h5_file['output_0'] = OneHot(downsampled_annotation_img_array)


def OneHot(annotaion_array):

    # One Hot
    # 背景, output : row x column x slices x 6
    output = np.zeros((annotaion_array.shape[0], annotaion_array.shape[1], 6))
    output[..., 0] = np.asarray(annotaion_array == 0, dtype=np.uint8) # save background
    output[..., 1] = np.asarray(annotaion_array == 1, dtype=np.uint8)
    output[..., 2] = np.asarray(annotaion_array == 3, dtype=np.uint8)
    output[..., 3] = np.asarray(annotaion_array == 4, dtype=np.uint8)
    output[..., 4] = np.asarray(annotaion_array == 5, dtype=np.uint8)
    output[..., 5] = np.asarray(annotaion_array == 6, dtype=np.uint8)

    return output


def CheckH5(h5_file_path):
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle(os.path.split(h5_file_path)[-1])

    plt.subplot(241)
    with h5py.File(h5_file_path, 'r') as target_h5_file:
        downsampled_core_img_array = target_h5_file['input_0'].value
        downsampled_annotation_img_array = target_h5_file['output_0'].value

    plt.imshow(downsampled_core_img_array)
    # plt.contour(downsampled_annotation_img_array)
    plt.title('input_0 ' +'\n'+str(downsampled_core_img_array.shape))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(242)
    plt.hist(downsampled_core_img_array.flatten(),normed=True)
    plt.title('pixel distribution of core img')


    plt.subplot(243)
    plt.imshow(downsampled_annotation_img_array[:, :, 0])
    plt.title('output_0 ' + '\n' + str(downsampled_annotation_img_array.shape)+'\n'+'000000')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(244)
    plt.imshow(downsampled_annotation_img_array[:, :, 1])
    plt.title('output_0 ' + '\n' + str(downsampled_annotation_img_array.shape)+'\n'+'010000')
    plt.xticks([])
    plt.yticks([])


    plt.subplot(245)
    plt.imshow(downsampled_annotation_img_array[:, :, 2])
    plt.title('output_0 ' + '\n'+str(downsampled_annotation_img_array.shape)+'\n' +'001000')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(246)
    plt.imshow(downsampled_annotation_img_array[:, :, 3])
    plt.title('output_0' +'\n'+str(downsampled_annotation_img_array.shape)+'\n'+'000100')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(247)
    plt.imshow(downsampled_annotation_img_array[:, :, 4])
    plt.title('output_0 ' +'\n'+str(downsampled_annotation_img_array.shape)+'\n'+'000010')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(248)
    plt.imshow(downsampled_annotation_img_array[:, :, 5])
    plt.title('output_0 ' +'\n'+str(downsampled_annotation_img_array.shape)+'\n'+'000001')
    plt.xticks([])
    plt.yticks([])


    plt.show()

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

# CheckH5(r'C:\Users\zj\Desktop\Gleason\slide001_core003_Maps1_T.h5')
Iteration(r'W:\MRIData\OpenData\Gleason2019\Train Imgs',
          r'W:\MRIData\OpenData\Gleason2019\Maps1_T', r'U:\PrcoessedData\Challenge_Gleason2019\ProcessedH5_Maps1')