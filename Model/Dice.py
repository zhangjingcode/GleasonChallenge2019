import numpy as np
import os
import h5py

import matplotlib.pyplot as plt

from CustomerPath import h5_folder
from GleasonChallenge2019.Model.MergePredLabel import MergeOnePred

from GleasonChallenge2019.Model.Generator import GetH5Data, ImageIn2DTest
from CNNModel.Training.Generate import ImageInImageOut2D


def Dice_Coef(y_true, y_pred):
    # parameter for loss function
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
    return dice


def Binary(label, threshold):
    binary_label = np.asarray(label > threshold, dtype=np.uint8)

    return binary_label


def Projection(one_label):
    merged_one_label = np.zeros(shape=one_label.shape)
    for raw in range(one_label.shape[0]):
        for colunms in range(one_label.shape[1]):
            index = np.argmax(one_label[raw, colunms])
            merged_one_label[raw, colunms, index] = 1
    return merged_one_label


def ShowDice(h5_folder):
    label_list, pred_list, _ = GetH5Data(h5_folder)
    dice = []
    for index in range(len(label_list)):
        sum_dice = 0
        label = label_list[index]
        predict = pred_list[index]
        one_dice = []
        for channel in range(label.shape[-1]):
            binary_pred = Binary(predict[:, :, channel], threshold=0.5)
            binary_label = Binary(label[:, :, channel], threshold=0.5)
            one_dice.append(Dice_Coef(binary_label, binary_pred))

            # plt.suptitle('One-hot ' + str(channel) + '\n' + str(Dice_Coef(binary_label, binary_pred)))
            #
            # plt.subplot(121)
            # plt.title('Label')
            # plt.imshow(binary_label)
            # plt.axis('off')
            #
            # plt.subplot(122)
            # plt.title('Pred')
            # plt.imshow(binary_pred)
            # plt.axis('off')
            #
            # plt.show()

            # save_path = r'D:\ZYH\Data\GleasonChallenge2019\Voted_10down\model\Dice'
            # sub_save_path = os.path.join(save_path, str(index))
            # if not os.path.exists(sub_save_path):
            #     os.makedirs(sub_save_path)
            # image_save_path = os.path.join(sub_save_path, str(channel) + '.jpg')
            # plt.savefig(image_save_path)
            # plt.close()
        sum_dice += sum(one_dice)
        dice.append(sum_dice/6)
        print(index)
    print(sum(dice)/len(label_list))
    plt.hist(dice)
    plt.title('dice')
    plt.show()


# ShowDice(r'D:\ZYH\Data\GleasonChallenge2019\Voted_10down\transmodel\PredictH5')


def ShowMergedDice(h5_folder):
    label_list, pred_list, _ = GetH5Data(h5_folder)
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


# ShowMergedDice(r'D:\ZYH\Data\GleasonChallenge2019\Voted_10down\transmodel\PredictH5')


def ShowTest(h5_folder):
    pred_list, case_name = GetH5Data(h5_folder)
    for index in range(len(pred_list)):

        predict = pred_list[index]
        pro_label = Projection(predict)

        plt.figure(figsize=(16, 8))
        plt.margins(0.2, 0.2)
        plt.suptitle(case_name[index])

        for channel in range(predict.shape[-1]):
            plt.subplot(2, 3, channel+1)
            if channel > 1:
                plt.title('GleasonScore' + str(channel+1))
            else:
                plt.title('GleasonScore' + str(channel))
            plt.imshow(pro_label[:, :, channel])
            plt.axis('off')

        plt.show()
        plt.close()


# ShowTest(r'D:\ZYH\Data\GleasonChallenge2019\Test_h5\transmodel\TestPredictH5')






