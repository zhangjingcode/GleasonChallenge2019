import numpy as np
import os
import h5py

import matplotlib.pyplot as plt

from CustomerPath import h5_folder
from GleasonChallenge2019.Model.PredictTest import MergeOnePred


def GetH5Data(data_folder):
    file_list = os.listdir(data_folder)
    label_list = []
    predict_list = []

    for file in file_list:
        file_path = os.path.join(data_folder, file)
        with h5py.File(file_path, 'r') as f:
            # print(f.keys())
            label = np.asarray(f['output_0'], dtype=float)
            predict = np.asarray(f['predict_0'], dtype=np.float32)

            label_list.append(label)
            predict_list.append(predict)

    return np.asarray(label_list), np.asarray(predict_list)


def Dice_Coef(y_true, y_pred):
    # parameter for loss function
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
    return dice


def Binary(label, threshold):
    binary_label = np.asarray(label > threshold, dtype=np.uint8)

    return binary_label


def ShowDice(thres):
    label_list, pred_list = GetH5Data(h5_folder)
    for index in range(len(label_list)):
        dice = []
        sum_dice = 0

        label = label_list[index]
        predict = pred_list[index]

        for channel in range(label.shape[-1]):
            binary_pred = Binary(predict[:, :, channel], threshold=1/6)
            binary_label = Binary(label[:, :, channel], threshold=1/6)
            dice.append(Dice_Coef(binary_label, binary_pred))

            plt.suptitle('One-hot ' + str(channel) + '\n' + str(Dice_Coef(binary_label, binary_pred)))

            plt.subplot(121)
            plt.title('Label')
            plt.imshow(binary_label)
            plt.axis('off')

            plt.subplot(122)
            plt.title('Pred')
            plt.imshow(binary_pred)
            plt.axis('off')

            # plt.show()

            save_path = r'D:\ZYH\Data\GleasonChallenge2019\Voted_10down\model\Dice'
            sub_save_path = os.path.join(save_path, str(index))
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)
            image_save_path = os.path.join(sub_save_path, str(channel) + '.jpg')
            plt.savefig(image_save_path)
            plt.close()
            sum_dice += sum(dice)
        print(sum(dice))
        print(sum(dice))


# ShowDice()


def ShowMergedDice():
    label_list, pred_list = GetH5Data(h5_folder)
    for index in range(len(label_list)):
        dice = []

        label = label_list[index]
        predict = pred_list[index]

        merged_label = MergeOnePred(label)
        merged_pred = MergeOnePred(predict)

        dice.append(Dice_Coef(merged_label, merged_pred))

        plt.suptitle('One-hot ' + '\n' + str(Dice_Coef(merged_label, merged_pred)))

        plt.subplot(121)
        plt.title('Label')
        plt.matshow(merged_label, fignum=0)
        plt.axis('off')

        plt.subplot(122)
        plt.title('Pred')
        plt.matshow(merged_pred, fignum=0)
        plt.axis('off')

        plt.show()

        print(dice)


# ShowMergedDice()