import numpy as np
import os
import h5py
import cv2

import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import ndimage


def MergeOnePred(one_label):
    merged_one_label = np.zeros(shape=(one_label.shape[0], one_label.shape[1]), dtype=np.float32)
    for raw in range(one_label.shape[0]):
        for colunms in range(one_label.shape[1]):
            index = np.argmax(one_label[raw, colunms])
            if index > 1:
                merged_one_label[raw, colunms] = index + 1
            else:
                merged_one_label[raw, colunms] = index

    return merged_one_label


def TestMergeOnePred(save_path, output_list):
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    one_label = output_list[0, :]
    one_predict = pred[0, :]
    merged_predict = MergeOnePred(one_predict)

    plt.subplot(121)
    plt.contour(one_label[:, :, 0], colors='r')
    plt.imshow(one_predict[:, :, 0], cmap='gray')
    plt.subplot(122)
    plt.contour(one_label[:, :, 0], colors='r')
    plt.imshow(merged_predict[:, :, 0], cmap='gray')

    plt.show()


# TestMergeOnePred()

def MergeLabel(save_path, pred):
    for case_num in range(pred.shape[0]):

        merged_pred = MergeOnePred(pred[case_num, :])

        # 中值滤波
        # merged_median_pred = signal.medfilt(merged_pred, 15)

        # binary_fill_holes()
        # merged_median_pred = FillHole_RGB(merged_pred, SavePath=False)

        plt.figure(figsize=(16, 8))
        plt.subplot(131)
        plt.axis('off')
        plt.title('pred')
        # plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(pred[case_num, :, :, 0], cmap='gray')

        plt.subplot(132)
        plt.axis('off')
        plt.title('merged_pred')
        # plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(merged_pred, cmap='gray')

        plt.subplot(133)
        plt.axis('off')
        plt.title('merged_median_pred')
        # plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(merged_median_pred, cmap='gray')
        plt.show()

        if save_path:
            sub_save_path = os.path.join(save_path, 'result_merged_label', str(case_num)+'.jpg')
            plt.savefig(sub_save_path)
            plt.close()


# from GleasonChallenge2019.Model.Generator import GetH5Data
# predict_list, _ = GetH5Data(r'D:\ZYH\Data\GleasonChallenge2019\Test_h5\model\ProjectPredictH5')
# MergeLabel(save_path=False, pred=predict_list)