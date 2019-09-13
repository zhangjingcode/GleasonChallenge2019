import os
import h5py
import numpy as np

from CNNModel.Training.Generate import _CropDataList2D, _AddOneSample, _MakeKerasFormat, GetKeysFromStoreFolder

def _GetInputNumber(case_folder):
    key_list = GetKeysFromStoreFolder(case_folder)

    input_number = 0
    for key in key_list:
        if 'input' in key:
            input_number += 1
        else:
            print(key)

    if input_number > 0:
        return input_number
    else:
        print('Lack input or output: ', case_folder)
        return 0, 0

def ImageIn2DTest(root_folder, input_shape):
    from MeDIT.Visualization import LoadWaitBar
    input_number = _GetInputNumber(root_folder)
    case_list = os.listdir(root_folder)
    case_list = [case for case in case_list if case.endswith('.h5')]

    input_list = [[] for index in range(input_number)]
    shape_list = []

    for case in sorted(case_list):
        LoadWaitBar(len(case_list), case_list.index(case))
        case_path = os.path.join(root_folder, case)

        input_data_list = []
        file = h5py.File(case_path, 'r')
        shape = tuple(file['original_size'])
        shape_list.append(shape)
        for input_number_index in range(input_number):
            input_data_list.append(np.asarray(file['input_' + str(input_number_index)]))
        file.close()

        input_data_list = _CropDataList2D(input_data_list, input_shape)

        _AddOneSample(input_list, input_data_list)

        inputs = _MakeKerasFormat(input_list)


    return inputs, shape_list, case_list
