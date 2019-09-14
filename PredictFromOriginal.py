import os
import copy

import cv2
import numpy as np
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


def ImgPreProcess(core_img_array, target_size=(512, 512)):

    original_size = core_img_array.shape()

    downsampled_core_img_array = cv2.resize(core_img_array, target_size, interpolation=cv2.INTER_CUBIC)

    return downsampled_core_img_array, original_size


def PredictPostProcess(predict_array, original_size):
    one_hot_index = np.argmax(predict_array, axis=2)
    one_hot_dict = [0, 1, 3, 4, 5, 6]
    final_array = copy.deepcopy(one_hot_index)
    final_array[np.where(one_hot_index > 1)] = one_hot_index[np.where(one_hot_index > 1)] + 1

    # final_one_hot_array = OneHot(final_array)
    enlargement_array = cv2.resize(final_array, original_size, interpolation=cv2.INTER_NEAREST)

    return predict_array, final_array, enlargement_array


def GetTestImg(core_img_path):
    test_img_dict = {}
    if os.path.isfile(core_img_path):
        core_img_array, case_name = ReadCoreImg(core_img_path)
        downsampled_core_img_array, original_size = ImgPreProcess(core_img_array)
        test_img_dict[case_name] = [downsampled_core_img_array, original_size]

    elif os.path.isdir(core_img_path):
        for sub_core_img in os.path.isdir(core_img_path):
            sub_core_img_path = os.path.join(core_img_path, sub_core_img)

            core_img_array, case_name = ReadCoreImg(sub_core_img_path)
            downsampled_core_img_array, original_size = ImgPreProcess(core_img_array)
            test_img_dict[case_name] = [downsampled_core_img_array, original_size]

    return test_img_dict

def TestModel(model_path, test_img_dict, store_path):
    from keras.models import model_from_yaml

    def LoadModel(store_folder, weight_name='last_weights.h5', is_show_summary=False):
        yaml_file = open(os.path.join(store_folder, 'model.yaml'), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights(os.path.join(store_folder, weight_name))

        if is_show_summary:
            model.summary()
        return model
    test_img_dict = sorted(test_img_dict)
    case_name_list = list(test_img_dict.keys())
    case_pred_array_list = [index[0] for index in test_img_dict.values]
    case_original_size_list = [index[1] for index in test_img_dict.values]

    for index in range(len(test_img_dict)):
        case_name = case_name_list[index]
        input_list = case_pred_array_list[index]
        original_size = case_original_size_list[index]


        model = LoadModel(model_path, 'best_weights.h5', is_show_summary=True)
        pred = model.predict(input_list, batch_size=1)

        predict_array, final_array, enlargement_array = PredictPostProcess(pred, original_size)

        plt.title(case_name)
        plt.imshow(enlargement_array)
        plt.savefig(os.path.join(store_path, case_name+'_predict.png'))
        plt.close()