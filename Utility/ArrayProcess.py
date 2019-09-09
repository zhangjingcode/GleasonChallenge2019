import numpy as np

from collections import Counter

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

def SeparateLabel(annotation_img_array):
    seperate_label_array_dict = {}

    # Separate 1, 3, 4, 5, 6 voxel array
    print(Counter(annotation_img_array.flatten()))
    for specific_label in [1, 3, 4, 5, 6]:
        if specific_label in annotation_img_array:
            sub_annotation_img_array = np.zeros(annotation_img_array.shape)
            sub_annotation_img_array[annotation_img_array == specific_label] = specific_label
            seperate_label_array_dict[specific_label] = sub_annotation_img_array
    return seperate_label_array_dict