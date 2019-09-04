# test
import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
import scipy.signal as signal

from CustomerPath import model_path, testing_folder, save_path
from CNNModel.Training.Generate import ImageInImageOut2DTest


input_shape = [496, 496, 3]
batch_size = 4


def LoadTest(testing_folder, input_shape):

    input_list, output_list, case_list = ImageInImageOut2DTest(testing_folder, input_shape=input_shape)
    return input_list, output_list, case_list


input_list, output_list, case_list = LoadTest(testing_folder, input_shape)


def SavePredict(model_path, input_list, save_path, batch_size):
    from CNNModel.Utility.SaveAndLoad import LoadModel
    model = LoadModel(model_path, 'best_weights.h5', is_show_summary=True)
    pred = model.predict(input_list, batch_size=batch_size)
    np.save(os.path.join(save_path, 'prediction_test.npy'), pred)


# SavePredict(model_path, input_list, save_path, batch_size)


def ShowPred(output_list, save_path=''):

    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))

    for case_num in range(output_list.shape[0]):

        plt.figure(figsize=(16, 8))
        plt.margins(0.2,0.2)
        plt.suptitle(str(case_num))
        plt.subplot(231)
        plt.contour(output_list[case_num, :, :, 0], colors='r')
        plt.imshow(pred[case_num, :, :, 0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 100000')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(232)
        plt.contour(output_list[case_num, :, :, 1], colors='r')
        plt.imshow(pred[case_num, :, :, 1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 010000')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(233)
        plt.contour(output_list[case_num, :, :, 2], colors='r')
        plt.imshow(pred[case_num, :, :, 2], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 001000')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(234)
        plt.contour(output_list[case_num, :, :, 3], colors='r')
        plt.imshow(pred[case_num, :, :, 3], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 000100')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(235)
        plt.contour(output_list[case_num, :, :, 4], colors='r')
        plt.imshow(pred[case_num, :, :, 4], cmap='gray', vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 000010')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(236)
        plt.contour(output_list[case_num, :, :, 5], colors='r')
        plt.imshow(pred[case_num, :, :, 5], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('OneHot 000001')
        plt.plot(0, 0, '-', color='r', label='Annotation')
        plt.legend()


        if save_path:
            sub_save_path = os.path.join(save_path, 'result', str(case_num)+'.jpg')
            plt.savefig(sub_save_path)
            plt.close()
        # plt.show()


# ShowPred(output_list, save_path=save_path)
# plt.savefig(os.path.join(os.path.split(model_path)[0], 'ROC.png'))


def SavePredH5(input, output, save_path=''):
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    for case_num in range(output.shape[0]):
        label_name = 'data' + str(case_num) + '.h5'
        data_path = os.path.join(save_path, 'PredictH5', label_name)
        with h5py.File(data_path, 'w') as f:
            f['input_0'] = input[case_num, :]
            f['output_0'] = output[case_num, :]
            f['predict_0'] = pred[case_num, :]
# SavePredH5(input_list, output_list, save_path)

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


def TestMergeOnePred():
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    one_label = output_list[0, :]
    one_predict = pred[0, :]
    merged_predict = MergeOnePred(one_predict)


    # array_median = cv2.medianBlur(merged_predict, 5)
    plt.subplot(121)
    plt.contour(one_label[:, :, 0], colors='r')
    plt.imshow(one_predict[:, :, 0], cmap='gray')
    plt.subplot(122)
    plt.contour(one_label[:, :, 0], colors='r')
    plt.imshow(merged_predict[:, :, 0], cmap='gray')

    plt.show()


# TestMergeOnePred()

def MergeLabel(save_path, show_pixls=False):
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    print()
    for case_num in range(pred.shape[0]):

        merged_pred = MergeOnePred(pred[case_num, :])


        # 中值滤波
        merged_median_pred = signal.medfilt(merged_pred, 15)

        plt.figure(figsize=(16, 8))
        plt.subplot(131)
        plt.axis('off')
        plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(pred[case_num, :, :, 0], cmap='gray')

        plt.subplot(132)
        plt.axis('off')
        plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(merged_pred, cmap='gray')

        plt.subplot(133)
        plt.axis('off')
        plt.contour(output_list[case_num, :, :, 0], color='r')
        plt.imshow(merged_median_pred, cmap='gray')

        if save_path:
            sub_save_path = os.path.join(save_path, 'result_merged_label', str(case_num)+'.jpg')
            plt.savefig(sub_save_path)
            plt.close()

        if show_pixls:
            pred_pixls = np.unique(merged_pred)
            print("name:{},     pred pixls:{}".format(case_list[case_num], pred_pixls))


# MergeLabel(save_path)


data_path = r'D:\data\GleasonChallenge2019\Merged_512\model\PredictH5\data10.h5'
with h5py.File(data_path, 'r') as file:
    image = np.asarray(file['input_0'], dtype=np.float32)
    label = np.asarray(file['output_0'], dtype=np.uint8)
    predict = np.asarray(file['predict_0'], dtype=np.float32)
    plt.contour(label[:, :, 0], colors='r')
    plt.imshow(predict[:, :, 0], cmap='gray')
    plt.show()

