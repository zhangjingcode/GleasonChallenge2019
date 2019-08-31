from collections import Counter
import cv2
import pandas as pd


def Test():
    label_path = r'Z:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core037_classimg_nonconvex.png'
    label = cv2.imread(label_path)
    info_dict = dict(Counter(label.flatten()))
    number = 0
    keys = list(info_dict.keys())
    Dict = {}
    for i in keys:
        number = number + info_dict[i]

    for i in keys:
        Dict[i] = info_dict[i] / number

    print(info_dict)
    print(Dict, sum)
    return 0
# Test()


def GetInfo():
    import os

    data_path = r'Z:\MRIData\OpenData\Gleason2019\Maps2_T'
    file_list = os.listdir(data_path)
    all_data = pd.DataFrame()
    sum_dict = {0: 0, 1: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for file in file_list:
        label_path = os.path.join(data_path, file)
        label = cv2.imread(label_path)
        info_dict = dict(Counter(label.flatten()))
        Keys = list(info_dict.keys())
        number = 0
        Dict = {}

        for i in Keys:
            number = number + info_dict[i]
        for i in Keys:
            Dict[i] = info_dict[i] / number
            if Dict[i] != 0:
                sum_dict[i] = sum_dict[i] + 1

        Data = pd.DataFrame(Dict, index=[file])
        all_data = pd.concat([all_data, Data])

        print(file)
    Sum = pd.DataFrame(sum_dict, index=['sum'])
    all_data = pd.concat([all_data, Sum])
    all_data = all_data.fillna(0)
    all_data.to_csv(r'C:\Users\Cherish\Desktop\DataInformation0.csv', mode='a',)

    return 0


GetInfo()

