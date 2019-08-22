import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
import pandas as pd


def test():
    label_path = r'Z:\MRIData\OpenData\Gleason2019\Maps1_T\slide001_core037_classimg_nonconvex.png'
    label = cv2.imread(label_path)
    info_dict = dict(Counter(label.flatten()))
    number = 0
    keys = list(info_dict.keys())
    Dict = {}
    for i in keys:
        number = number + info_dict[i]
    for i in keys:
        Dict = {i: info_dict[i] / number}
    return Dict
# Dict = test()

def GetInfo():
    import os

    data_folder = r'Z:\MRIData\OpenData\Gleason2019\Maps1_T'
    file_list = os.listdir(data_folder)
    AllData = pd.DataFrame()
    for file in file_list:
        label_path = os.path.join(data_folder, file)
        label = cv2.imread(label_path)
        info_dict = dict(Counter(label.flatten()))
        number = 0
        keys = list(info_dict.keys())
        Dict = {}

        for i in keys:
            number = number + info_dict[i]
        for i in keys:
            Dict[i] = info_dict[i] / number
        print(file)

        data = pd.DataFrame(Dict, index=[file],)
        AllData = pd.concat([AllData, data])
        AllData = AllData.fillna(0)
        print(file)
    AllData.to_csv(r'C:\Users\Cherish\Desktop\DataInformation.csv', mode='a',)


    return 0


GetInfo()

