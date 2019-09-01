import os

from collections import Counter
import cv2
import pandas as pd

from CustomerPath import label_1_statistic_path

def StatisticOneLabel(label_path):
    label = cv2.imread(label_path)
    info_dict = dict(Counter(label.flatten()))
    total_voxel_number = label.size

    percentage_per_label = {}
    for i in info_dict.keys():
        percentage_per_label[i] = info_dict[i] / total_voxel_number

    print(info_dict)
    return percentage_per_label

def TestatisticOneLabel():
    label_path = r'Z:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core037_classimg_nonconvex.png'
    StatisticOneLabel(label_path)


# TestatisticOneLabel()


def StatisticLabel(folder_path, store_path):
    all_label_df = pd.DataFrame()
    for file in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, file)
        percentage_per_label = StatisticOneLabel(label_path)

        one_label_df = pd.DataFrame(percentage_per_label, index=[file])
        all_label_df = pd.concat([all_label_df, one_label_df])
        print(file)

    all_label_df = all_label_df.fillna(0)
    all_label_df.to_csv(store_path, mode='a')

def Demo():
    df = pd.read_csv(label_1_statistic_path, header=0, index_col=0)
    result = (df > 0.0).sum(axis=0)
    print(result)

# Demo()


def GetLabelShape(folder_path, store_path):
    all_shape_df = pd.DataFrame()
    for file in sorted(os.listdir(folder_path)):
        shape_dict = {}
        label_path = os.path.join(folder_path, file)
        label = cv2.imread(label_path)
        shape_dict['rows'] = label.shape[0]
        shape_dict['columns'] = label.shape[1]
        shape_dict['channels'] = label.shape[-1]
        print(file)
        label_df = pd.DataFrame(shape_dict, index=[file])
        all_shape_df = pd.concat([all_shape_df, label_df])
    all_shape_df.to_csv(store_path, mode='a')


from CustomerPath import folder_path, store_path
# GetLabelShape(folder_path, store_path)