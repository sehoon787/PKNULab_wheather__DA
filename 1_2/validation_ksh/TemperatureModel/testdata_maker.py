from __init__ import extract_col

import pandas as pd


class TestDataMaker:
    def __init__(self, path='./drive/My Drive/Colab Notebooks/공모전_0729/res.csv'):
        self.extract_col = extract_col

        self.path = path

        self.zeroTo = 0.00001
        self.missingCond = -999

    def load_data(self, path):
        return pd.read_csv(path, encoding="utf-8-sig")[self.extract_col]

    def sort_data(self, df, mode, sort_by=['YearMonthDayHourMinute', 'STN']):
        df_rt = pd.DataFrame()
        if mode == "inv":
            df_rt = df.sort_values(by=sort_by)
            df_rt['LandType'] += 1
            df_rt = df_rt.replace(0, self.zeroTo)
        elif mode == "rev":
            df_rt = df.sort_values(by=sort_by)

        return df_rt

    def pr_msv(self, list_):
        for i in list_:
            if i is not self.missingCond:
                first = i
                break

        pre = first
        list_rt = []

        for i, item in enumerate(list_):
            if item is not self.missingCond:
                list_rt.append(item)
                pre = item
            else:
                list_rt.append((pre + item) / 2)

        return list_rt

    def run(self, df, res_path=""):
        df_sort = self.sort_data(df, "inv")

        df_impu = pd.DataFrame()
        for (column, data) in df_sort.iteritems():
            df_impu[column] = self.pr_msv(data.values)

        df_final = self.sort_data(df_impu, "rev")
        df_final.to_csv(res_path + "/answer.csv", encoding='utf-8-sig', index=False)

        return df_final
