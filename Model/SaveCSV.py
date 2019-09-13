import os

import cv2
import pandas as pd

from GleasonChallenge2019.Model.GleasonScore import OneGleasonScore
from GleasonChallenge2019.Model.Dice import GetH5Data
from CustomerPath import label_folder


def SaveCSV(folder, store_path):
    predict_list, file_list = GetH5Data(folder)
    all_label_df = pd.DataFrame()
    for index in range(len(predict_list)):
        predict = predict_list[index]
        score_dict = OneGleasonScore(predict)
        del score_dict[0], score_dict[1], score_dict[6]
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]))
        i = 0
        score = []
        for keys in reversed(list(score_dict.keys())):
            i += 1
            if score_dict[keys]:
                score.append(keys)
            if i == 2:
                break

        one_label_df = pd.DataFrame(str(score), index=[file_list[index]], columns=['GleasonScore'])
        all_label_df = pd.concat([all_label_df, one_label_df])
        print(file_list[index])
        print(score)

    all_label_df.to_csv(store_path)



SaveCSV(label_folder, r'C:\Users\Cherish\Desktop\GleasonScore.csv')