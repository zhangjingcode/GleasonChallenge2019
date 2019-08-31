# test


model_path = r'D:\Gleason2019\TrainValidationTest_256\model_binary_entropy'
from MeDIT.SaveAndLoad import LoadModel
model = LoadModel(model_path, 'best_weights.h5', is_show_summary=True)

# training_folder = r'E:\ProcessData\ImageQuality-Checked-ZJ-SY\training'
# validation_folder = r'E:\ProcessData\ImageQuality-Checked-SY\validation'
testing_folder = r'D:\Gleason2019\TrainValidationTest_256\Test'

input_shape = [240, 240, 3]
batch_size = 4

from CNNModel.Training.Generate import ImageInImageOut2DTest

input_list, output_list, case_list = ImageInImageOut2DTest(testing_folder, input_shape=input_shape)
#(23, 240, 240, 6)
pred = model.predict(input_list, batch_size=batch_size)


def ShowPred(output_list, pred, save_path=''):
    import matplotlib.pyplot as plt
    import numpy as np

    for case_num in range(output_list.shape[0]):

        # for one_hot_index in range(output_list.shape[-1]):
        #     plt.contour(output_list[case_num, :, :, one_hot_index], colors='blue')
        #     cs = plt.imshow(pred[case_num, :, :, one_hot_index], cmap='gray')
        #     plt.colorbar(cs)
        #     plt.title(str(case_num)+'_'+str(one_hot_index))
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.show()
        #     plt.close()
        # plt.close()
        plt.figure(figsize=(16, 8))
        plt.margins(0.2,0.2)
        plt.suptitle(str(case_num))
        plt.subplot(231)
        plt.contour(output_list[case_num, :, :, 0], colors='r')
        plt.imshow(pred[case_num, :, :, 0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 100000')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

        #
        plt.subplot(232)
        plt.contour(output_list[case_num, :, :, 1], colors='r')
        plt.imshow(pred[case_num, :, :, 1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 010000')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()
        #
        plt.subplot(233)
        plt.contour(output_list[case_num, :, :, 2], colors='r')
        plt.imshow(pred[case_num, :, :, 2], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 001000')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(234)
        plt.contour(output_list[case_num, :, :, 3], colors='r')
        plt.imshow(pred[case_num, :, :, 3], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 000100')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(235)
        plt.contour(output_list[case_num, :, :, 4], colors='r')
        plt.imshow(pred[case_num, :, :, 4], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 000010')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(236)
        plt.contour(output_list[case_num, :, :, 5], colors='r')
        plt.imshow(pred[case_num, :, :, 5], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 000001')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()



        if save_path:
            import os
            sub_save_path = os.path.join(save_path, str(case_num)+'.jpg')
            plt.savefig(sub_save_path)
            plt.close()
        # plt.show()
ShowPred(output_list, pred, save_path=r'D:\Gleason2019\TrainValidationTest_256\model_binary_entropy\PredShowTest')
# plt.savefig(os.path.join(os.path.split(model_path)[0], 'ROC.png'))



