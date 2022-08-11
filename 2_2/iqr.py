import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

extract_col = ['year', 'month', 'day',
'3시간기온_4', '3시간기온_7', '3시간기온_10', '3시간기온_13', '3시간기온_16', '3시간기온_19',
'3시간기온_22', '3시간기온_25', '3시간기온_28', '3시간기온_31', '3시간기온_34', '3시간기온_37',
'3시간기온_40', '3시간기온_43', '3시간기온_46', '3시간기온_49', '3시간기온_52', '3시간기온_55',
'강수형태_4', '강수형태_7', '강수형태_10', '강수형태_13', '강수형태_16', '강수형태_19',
'강수형태_22', '강수형태_25', '강수형태_28', '강수형태_31', '강수형태_34', '강수형태_37',
'강수형태_40', '강수형태_43', '강수형태_46', '강수형태_49', '강수형태_52', '강수형태_55',
'강수확률_4', '강수확률_7', '강수확률_10', '강수확률_13', '강수확률_16', '강수확률_19',
'강수확률_22', '강수확률_25', '강수확률_28', '강수확률_31', '강수확률_34', '강수확률_37',
'강수확률_40', '강수확률_43', '강수확률_46', '강수확률_49', '강수확률_52', '강수확률_55',
'습도_4', '습도_7', '습도_10', '습도_13', '습도_16', '습도_19',
'습도_22', '습도_25', '습도_28', '습도_31', '습도_34', '습도_37',
'습도_40', '습도_43', '습도_46', '습도_49', '습도_52', '습도_55',
'일최고기온_4', '일최고기온_7', '일최저기온_4', '일최저기온_7',
'pop_ratio', 'man_pop', 'women_pop',
 'man_0~4', 'man_5~9', 'man_10~14', 'man_15~19', 'man_20~24', 'man_25~29',
'man_30~34', 'man_35~39', 'man_40~44', 'man_45~49', 'man_50~54', 'man_55~59',
'man_60~64', 'man_65~69', 'man_70~74', 'man_75~79', 'man_80~84', 'man_85~89',
'man_90~94', 'man_95~99', 'man_100 이상',
'woman_0~4', 'woman_5~9', 'woman_10~14', 'woman_15~19', 'woman_20~24',
'woman_25~29', 'woman_30~34', 'woman_35~39', 'woman_40~44', 'woman_45~49',
'woman_50~54', 'woman_55~59', 'woman_60~64', 'woman_65~69', 'woman_70~74',
'woman_75~79', 'woman_80~84', 'woman_85~89', 'woman_90~94', 'woman_95~99',
'woman_100 이상',
'gender', 'frequency']


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

    def run(self, remove_col=[], save_plot=False, draw_plot=False, encoding="utf-8-sig"):
        return self.check_outlier(self.path, remove_col=remove_col, save_plot=save_plot, draw_plot=draw_plot, encoding=encoding)

IQR("result4_train.csv").run(draw_plot=True)