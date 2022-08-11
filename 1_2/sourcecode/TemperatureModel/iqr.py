from __init__ import extract_col

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def outlier_iqr(data):
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)
    iqr = q75 - q25

    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    print('IQR은', iqr, '이다.')
    print('lower bound 값은', lower, '이다.')
    print('upper bound 값은', upper, '이다.')

    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기
    data1 = data[data > upper]
    data2 = data[data < lower]

    # 이상치 총 개수 구하기
    print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.\n')
    return lower, upper


class IQR:
    def __init__(self, path, res_path="iqr"):
        '''
        :param path: load file path
        :param res_path: result save path
        '''

        self.extract_col = extract_col

        self.path = path
        self.res_path = res_path

        self.upper = float('inf')
        self.lower = float('-inf')

    def _load_data(self, path, encoding="utf-8"):
        return pd.read_csv(path, encoding=encoding)[self.extract_col]

    def _draw_outlier(self, df, target, save_plot, draw_plot):
        try:
            plt.plot(df[target])
            plt.plot(df.loc[(df[target] > self.lower) & (df[target] < self.upper)][target])
            plt.plot(df.loc[(df[target] <= self.lower) | (df[target] >= self.upper)]['insitu-TG'])
            plt.suptitle(target)
            if draw_plot:
                plt.show()
            if save_plot:
                plt.savefig(self.res_path + "/" + target + ".jpg")
            plt.clf()
        except Exception as e:
            print(e)
            plt.clf()

    def check_outlier(self, path, remove_col=[], save_plot=False, draw_plot=False, encoding="utf-8"):
        os.makedirs(self.res_path, exist_ok=True)

        df = self._load_data(path)
        outllier_list = {}

        print("Total File Rows: " + str(len(df)))

        for target in tqdm(self.extract_col):
            print("IQR target: " + target)

            try:
                self.lower, self.upper = outlier_iqr(df[target])
                df = df.loc[(df[target] > self.lower) & (df[target] < self.upper)]

                outllier_list[target] = len(df) - (len(df.loc[(df[target] > self.lower) & (df[target] < self.upper)]))

                if len(remove_col) < 1:
                    df = df.loc[(df[target] > self.lower) & (df[target] < self.upper)]
                else:
                    if target in remove_col:
                        df = df.loc[(df[target] > self.lower) & (df[target] < self.upper)]
                print("Current Rows: " + str(len(df)))

                if save_plot | draw_plot:
                    self._draw_outlier(df=df, target=target, save_plot=save_plot, draw_plot=draw_plot)

            except Exception as e:
                print(e)

        if save_plot | draw_plot:
            plt.bar(outllier_list.keys(), outllier_list.values(), 1.0, color='g')
            plt.suptitle("Outlier Histogram")
            if draw_plot:
                plt.show()
            if save_plot:
                plt.savefig(self.res_path + "iqr.jpg")

        outlier_info_df = pd.DataFrame.from_dict([outllier_list])
        outlier_info_df.to_csv(self.res_path + "/outlier_info.csv", encoding=encoding, index=False)
        df.to_csv(self.res_path + "/IQR_removed_outlier.csv", encoding=encoding, index=False)

        return df, outlier_info_df, (self.res_path + "/IQR_removed_outlier.csv")

    def run(self, remove_col=[], save_plot=False, draw_plot=False, encoding="utf-8"):
        return self.check_outlier(self.path, remove_col=remove_col, save_plot=save_plot, draw_plot=draw_plot, encoding=encoding)