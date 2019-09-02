# test
<<<<<<< Updated upstream:PredictTest.py
import os
import numpy as np
import h5py
=======
import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.3 # don't hog all vRAM
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)
>>>>>>> Stashed changes:Test.py

import matplotlib.pyplot as plt

<<<<<<< Updated upstream:PredictTest.py
from CustomerPath import model_path, testing_folder, save_path
=======
<<<<<<< Updated upstream
model_path = r'D:\Gleason2019\TrainValidationTest_256\model_binary_entropy'
from MeDIT.SaveAndLoad import LoadModel
=======

model_path = r'D:\Gleason2019\TrainValidationTest_256\model_categorical_crossentropy'
from MeDIT.CNNModel.SaveAndLoad import LoadModel,LoadHistory
>>>>>>> Stashed changes
model = LoadModel(model_path, 'best_weights.h5', is_show_summary=True)
>>>>>>> Stashed changes:Test.py


input_shape = [240, 240, 3]
batch_size = 4

def LoadTest():
    from CNNModel.Training.Generate import ImageInImageOut2DTest
    input_list, output_list, case_list = ImageInImageOut2DTest(testing_folder, input_shape=input_shape)
    return input_list, output_list, case_list


<<<<<<< Updated upstream:PredictTest.py
def SavePredict(model_path, input_list, save_path, batch_size):
    from CNNModel.Utility.SaveAndLoad import LoadModel
    model = LoadModel(model_path, 'best_weights.h5', is_show_summary=True)
    pred = model.predict(input_list, batch_size=batch_size)
    np.save(os.path.join(save_path, 'prediction_test.npy'), pred)
=======
def SavePredArray(pred, save_path=''):
    import numpy as np
    import os
    for case_num in range(pred.shape[0]):
        pred_array = pred[case_num, :,:,:]
        np.save(os.path.join(save_path, str(case_num)+'.npy'), pred_array)

>>>>>>> Stashed changes:Test.py


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
        plt.title('OneHot 100000')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

        plt.subplot(232)
        plt.contour(output_list[case_num, :, :, 1], colors='r')
        plt.imshow(pred[case_num, :, :, 1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('OneHot 010000')
        plt.plot(0,0,'-', color='r', label='Annotation')
        plt.legend()

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
        plt.imshow(pred[case_num, :, :, 4], cmap='gray', vmin=0.0, vmax=1.0)
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
            sub_save_path = os.path.join(save_path, 'result', str(case_num)+'.jpg')
            plt.savefig(sub_save_path)
            plt.close()
        # plt.show()
<<<<<<< Updated upstream:PredictTest.py

# ShowPred(output_list, pred, save_path=r'D:\Gleason2019\TrainValidationTest_256\model_binary_entropy\PredShowTest')
=======
# LoadHistory(model_path, is_show=True)
# show_train_history(history, 'loss', 'val_loss')
# ShowPred(output_list, pred, save_path=r'D:\Gleason2019\TrainValidationTest_256\model_categorical_crossentropy\PredShow')
>>>>>>> Stashed changes:Test.py
# plt.savefig(os.path.join(os.path.split(model_path)[0], 'ROC.png'))
SavePredArray(pred, r'D:\Gleason2019\TrainValidationTest_256\TrainValidationTest_256\TestPred')

def SavePredH5(input, output, save_path=''):
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    for case_num in range(output.shape[0]):
        label_name = 'date' + str(case_num) + '.h5'
        data_path = os.path.join(save_path, 'LabelH5', label_name)
        with h5py.File(data_path, 'w') as f:
            f['input_0'] = input[case_num, :]
            f['output_0'] = output[case_num, :]
            f['predict_0'] = pred[case_num, :]


def MergeOneLabel(save_path):
    pred = np.load(os.path.join(save_path, 'prediction_test.npy'))
    one_label = pred[4, :]
    new_array = np.zeros(shape=(one_label.shape[0], one_label.shape[1], 1))
    for raw in range(one_label.shape[0]):
        for colunms in range(one_label.shape[1]):
            index = np.argmax(one_label[raw, colunms, :])
            if index > 2:
                new_array[raw, colunms, 0] = index + 1
            else:
                new_array[raw, colunms, 0] = index

    return new_array


def TestMergeOneLabel():
    import matplotlib.pyplot as plt
    array = MergeOneLabel(save_path)
    plt.imshow(array[:, :, 0], cmap='gray')
    plt.show()


TestMergeOneLabel()