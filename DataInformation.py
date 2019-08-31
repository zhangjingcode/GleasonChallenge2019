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

def TStatisticOneLabel():
    # label_path = r'Z:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core037_classimg_nonconvex.png'
    # StatisticOneLabel(label_path)
    pass

def GetInfo(folder_path, store_path):
    all_label_df = pd.DataFrame()
    sum_dict = {0: 0, 1: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for file in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, file)
        percentage_per_label = StatisticOneLabel(label_path)

        one_label_df = pd.DataFrame(percentage_per_label, index=[file])
        all_label_df = pd.concat([all_label_df, one_label_df])
        print(file)

    all_label_df = all_label_df.fillna(0)

    # Sum = pd.DataFrame(sum_dict, index=['sum'])
    # all_data = pd.concat([all_data, Sum])
    all_label_df.to_csv(store_path, mode='a')

def Demo():
    df = pd.read_csv(label_1_statistic_path, header=0, index_col=0)
    result = (df > 0.0).sum(axis=0)
    print(result)

Demo()

