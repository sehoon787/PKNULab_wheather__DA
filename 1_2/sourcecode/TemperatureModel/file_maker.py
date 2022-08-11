from __init__ import colnames, extract_col

import glob
import os
import pandas as pd
from tqdm import tqdm


class FileMaker:
    def __init__(self, dir_path="", res_path=""):
        self.colnames = colnames
        self.extract_col = extract_col

        self.dir_path = dir_path
        self.res_path = res_path

        self.zeroTo = 0.00001
        self.missingCond = -999

    def _dropMissingVal(self, load_path):
        try:
            df = pd.read_csv(load_path, names=self.colnames, header=None, usecols=self.colnames).dropna(axis=0)

            # LandType: 1:습지, 2: 산림 3: 늪지, 4: 도시, 5: 해안
            df['LandType'] += 1         # => 기존 LandType의 카테고리에서 1을 더함. 0에 가중치가 곱해지는 것을 방지하기 위함

            # 0 value to 0.00001
            df = df.replace(0, self.zeroTo)

            # for loop in colnames or extract_col
            for i in self.extract_col:
                df = df[(df[i] != self.missingCond)]

            return df

        except Exception as e:
            print(e)
            return False

    def makeDailyFile(self, load_path, target_path, remove_month=[]):
        try:
            for monthly_dir in os.listdir(load_path):
                if monthly_dir[-2:] in remove_month:
                    pass
                else:
                    result_path = target_path + "/" + monthly_dir
                    if not(os.path.isdir(result_path)):
                        os.makedirs(result_path, exist_ok=True)

                    monthly_path = load_path + "/" + monthly_dir
                    for daily_file in tqdm(os.listdir(monthly_path)):
                        df = self._dropMissingVal(load_path=monthly_path + "/" + daily_file)
                        if len(df) > 0:
                            df.to_csv(result_path + "/" + daily_file, index=False)

            print("-----------------------------------------------------------------------------------")
            print("-------------------------- Complete Delete Empty Values  --------------------------")
            print("-----------------------------------------------------------------------------------")

            return True

        except Exception as e:
            print(e)
            return False

    def makeMonthlyFile(self, load_path, target_path, remove_month=[]):
        if self.makeDailyFile(load_path=load_path, target_path=target_path, remove_month=remove_month):
            result_path = target_path + "/merge"
            os.makedirs(result_path, exist_ok=True)

            try:
                for monthly_dir in tqdm(os.listdir(target_path)):
                    if monthly_dir[-2:] in remove_month:
                        pass
                    else:
                        files_joined = os.path.join(target_path + "/" + monthly_dir, "*.LST.csv")

                        # Return a list of all joined files
                        list_files = glob.glob(files_joined)

                        # Merge files by joining all files
                        df = pd.concat(map(pd.read_csv, list_files), ignore_index=True)[self.extract_col]

                        # Attach month column
                        df['month'] = pd.to_datetime(df['YearMonthDayHourMinute'], format='%Y%m%d%H%M').dt.strftime('%m').astype(int)


                        # # Attach time column
                        # df['time'] = pd.to_datetime(df['YearMonthDayHourMinute'], format='%Y%m%d%H%M').dt.strftime('%H%M').astype(int)
                        #
                        # df = df.replace(0, self.zeroTo)

                        df.to_csv(target_path + "/merge/" + monthly_dir + "_merge.csv", index=False)
            except Exception as e:
                print(monthly_dir)
                print(e)
                return False

            print("-----------------------------------------------------------------------------------")
            print("-------------------------- Complete Merge Each Month Files ------------------------")
            print("-----------------------------------------------------------------------------------")

            return True

        return False

    def mergeFile(self, load_path, target_path, fileName="merge_data(1)", resampling=True, remove_month=[]):
        res_path = target_path + "/" + fileName + ".csv"

        try:
            if resampling:
                if not self.makeMonthlyFile(load_path, target_path, remove_month=remove_month):
                    False

            with open(res_path, 'w') as f:
                for file in tqdm(os.listdir(target_path + "/merge")):
                    f.write(self.extract_col)
                    with open(target_path + "/merge/" + file, 'r') as f2:
                        lines = f2.readlines()  # 2.merge 대상 파일의 row 1줄을 읽어서
                        for line in lines[1:]:
                            f.write(line)  # 3.읽은 row 1줄을 merge할 파일에 쓴다.

            return res_path

        except Exception as e:
            print(e)
            return False

    def run(self):
        return self.mergeFile(load_path=self.dir_path, target_path=self.res_path)

