import os
import copy

import matplotlib.pyplot as plt
import h5py
import numpy as np
from collections import Counter

from ArrayProcess import SeparateLabel, OneHot

def MarginPlot(core_img_array='', annotation_img_array='', title='', show=False):
    """

    :param core_img_array: the array of img want to show
    :param annotation_img_array: the array of annotation want to show
    :param title:
    :return: ax object
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # show img
    if core_img_array != '':
        ax.imshow(core_img_array)

    if annotation_img_array != '':
        # set color

        color_map = {1: 'lawngreen', 3: 'darkorange', 4: 'blue', 5: 'red', 6: 'black'}

        # separate label array

        separate_label_array_dict = SeparateLabel(annotation_img_array)

        # show contour in different voxel threshold

        for voxel_threshold in [1, 3, 4, 5, 6]:
            if voxel_threshold in separate_label_array_dict.keys():
                seperate_label_array = separate_label_array_dict[voxel_threshold]
                ax.contour(seperate_label_array, colors=color_map[voxel_threshold], linewidths=1, linestyles='dotted')
                # add special label for margin
                ax.plot(0, 0, '-', label=voxel_threshold, color=color_map[voxel_threshold])

        # cs = ax.contour(annotation_img_array[:,:,0], levels=[1, 3, 4, 5, 6],
        #             cmap="Accent", linewidths=5)
        # fig.colorbar(cs, ax=ax,extendfrac=False)

        ax.legend()
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if show:
            plt.show()
    return ax


def ShowOneHot(core_img_array, modal_merged_annotation_array, vim=0, color_map='jet', store_path='', title='',show=True):
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    plt.suptitle(title)

    plt.subplot(241)
    plt.imshow(core_img_array)
    plt.xticks([])
    plt.yticks([])
    plt.title('Core Img' + '\n' + str(core_img_array.shape))

    plt.imshow(core_img_array)
    # plt.contour(downsampled_annotation_img_array)
    plt.title('input_0 ' + '\n' + str(modal_merged_annotation_array.shape))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(242)
    plt.hist(core_img_array.flatten(), density=True)
    plt.title('pixel distribution of core img')

    plt.subplot(243)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 0], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0 ' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '000000')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(244)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 1], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0 ' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '010000')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(245)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 2], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0 ' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '001000')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(246)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 3], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '000100')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(247)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 4], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0 ' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '000010')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(248)
    sc = plt.imshow(modal_merged_annotation_array[:, :, 5], vmin=vim, vmax=1, cmap=color_map)
    plt.colorbar(sc)
    plt.title('output_0 ' + '\n' + str(modal_merged_annotation_array.shape) + '\n' + '000001')
    plt.xticks([])
    plt.yticks([])

    if store_path:
        plt.savefig(store_path)

    if show:
        plt.show()

    plt.close()

def ShowH5(h5_file_path,store_path='', show=True):
    with h5py.File(h5_file_path, 'r') as target_h5_file:
        keys_index = list(target_h5_file.keys())
        core_img_array = target_h5_file['input_0'].value
        annotation_array = target_h5_file['output_0'].value

    ShowOneHot(core_img_array, annotation_array, store_path=store_path, show=show,
               vim=-1, color_map='cool',title=os.path.split(h5_file_path)[-1])


def ShowPredictH5(h5_file_path, store_path='', show=True):
    with h5py.File(h5_file_path, 'r') as target_h5_file:

        core_img_array = target_h5_file['input_0'].value
        annotation_array = target_h5_file['output_0'].value
        predict_img_array = target_h5_file['predict_0'].value

    # ShowOneHot(core_img_array, annotation_array, store_path=store_path, show=show,
    #            title=os.path.split(h5_file_path)[-1])
    ShowOneHot(core_img_array, predict_img_array-annotation_array,
               vim=-1, color_map='jet',
               store_path=store_path, show=show,
               title=os.path.split(h5_file_path)[-1]+' Predict - Core')

def ShowTestH5(test_h5_path):
    with h5py.File(test_h5_path, 'r') as target_h5_file:
        keys_index = list(target_h5_file.keys())
        core_img_array = target_h5_file['input_0'].value
        predict_array = target_h5_file['predict_0'].value

    one_hot_index = np.argmax(predict_array, axis=2)
    one_hot_dict = [0, 1, 3, 4, 5, 6]
    final_array = copy.deepcopy(one_hot_index)
    final_array[np.where(one_hot_index > 1)] = one_hot_index[np.where(one_hot_index > 1)] + 1

    final_one_hot_array = OneHot(final_array)


    return core_img_array, predict_array, final_array, final_one_hot_array



def main():
    from ReadAndSave import ReadLabelingImg
    #
    # annotation_array, annotation_name = \
    #     ReadLabelingImg(r'W:\MRIData\OpenData\Gleason2019\Maps2_T\slide002_core041_classimg_nonconvex.png')
    #
    # MarginPlot(annotation_img_array=annotation_array, title='Maps2_T\slide002_core041', show=True)

    test_h5_path= r'G:\GleasonChallenge2019\ProjectPredictH5\slide001_core015_.h5'
    core_img_array, predict_array, final_array,final_one_hot_array = ShowTestH5(test_h5_path)
    ShowOneHot(core_img_array, final_one_hot_array)
    # MarginPlot(core_img_array=core_img_array, annotation_img_array=final_array, show=True)
    # pred_path = r'C:\Users\zj\Desktop\word_and_ppt\GleasonChallenge\OneHotVote\transfer\PredictH5'
    # for index in os.listdir(pred_path):
    #     ShowPredictH5(os.path.join(pred_path, index))
if __name__ == "__main__":
    main()