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


def EnlargementPatch(patch_array, enlarged_size, center_point=[-1, -1], is_shift=True):

    '''
    Extract patch from a 2D image.
    :param patch_array: the size of the 2D patch, (448, 448)
    :param enlarged_size: the 2D numpy array, (5120//10, 4608//10)
    :param center_point: the center position of the patch
    :param is_shift: If the patch is too close to the edge of the image, is it allowed to shift the patch in order to
    ensure that extracting the patch close to the edge. Default is True.
    :return: the extracted patch.
    '''

    enlarged_array = np.zeros(enlarged_size)

    # if patch_array.shape == () or patch_array.shape == (1,):
    #     patch_array = np.array([patch_array[0], patch_array[1]])

    image_row = enlarged_size[0]
    image_col = enlarged_size[1]

    if patch_array.shape[0] // 2 == image_row - (patch_array.shape[0] // 2):
        catch_x_index = [patch_array.shape[0] // 2]
    else:
        catch_x_index = np.arange(patch_array.shape[0] // 2, image_row - (patch_array.shape[0] // 2))
    if patch_array.shape[1] // 2 == image_col - (patch_array.shape[1] // 2):
        catch_y_index = [patch_array.shape[1] // 2]
    else:
        catch_y_index = np.arange(patch_array.shape[1] // 2, image_col - (patch_array.shape[1] // 2))

    if center_point == [-1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2

    # if patch_array.shape[0] > image_row or patch_array.shape[1] < image_col:
    #     print('The patch_size is larger than image shape')
    #     return np.array([])

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return []
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return []
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return []
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return []

    patch_row_index = [center_point[0] - patch_array.shape[0] // 2, center_point[0] + patch_array.shape[0] - patch_array.shape[0] // 2]
    patch_col_index = [center_point[1] - patch_array.shape[1] // 2, center_point[1] + patch_array.shape[1] - patch_array.shape[1] // 2]

    enlarged_array[patch_row_index[0]:patch_row_index[1], patch_col_index[0]:patch_col_index[1]] = patch_array
    return enlarged_array, [patch_row_index, patch_col_index]

def CheckEnlargment(h5_file_path):
    import h5py
    with h5py.File(h5_file_path, 'r') as target_h5_file:

        core_img_array = target_h5_file['input_0'].value
        # annotation_array = target_h5_file['output_0'].value
        predict_img_array = target_h5_file['predict_0'].value

    EnlargementPatch(predict_img_array[..., 0], (512, 460))

def TestEnlarge():
    from GleasonChallenge2019.Model.Generator import ImageIn2DTest
    from GleasonChallenge2019.Model.Generator import GetH5Data

    import matplotlib.pyplot as plt

    input_shape = [448, 448, 3]

    input_list, shape_list, case_list = ImageIn2DTest(r'D:\ZYH\Data\GleasonChallenge2019\Test_h5\Test', input_shape=input_shape)
    predict_list, _ = GetH5Data(r'D:\ZYH\Data\GleasonChallenge2019\Test_h5\model\ProjectPredictH5')

    for case_num in range(len(case_list)):
        patch = (shape_list[case_num][0]//10, shape_list[case_num][1]//10)
        predict = predict_list[case_num]

        for index in range(predict.shape[-1]):
            enlarged_array, list = EnlargementPatch(predict[:, :, index], patch)

            print(list)
            plt.imshow(enlarged_array)
            plt.axis('off')
            plt.show()


def main():
    h5_path = r'D:\ZYH\Data\GleasonChallenge2019\Test_h5\model\ProjectPredictH5\slide001_core015_.h5'
    CheckEnlargment(h5_path)

if __name__ == '__main__':
    main()
# TestEnlarge()

