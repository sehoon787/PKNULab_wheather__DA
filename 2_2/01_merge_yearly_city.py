# -*- coding: utf-8 -*-
from __init__ import *

from datetime import datetime, timedelta
from glob import glob
import os
import pandas as pd
import platform
from tqdm import tqdm
import unicodedata


seperator = "\\" if "Window" in platform.platform() else "/"

PATH = "data/예보데이터/*"
COLUMNS = ["date", "year", "month", "day", "hour", "forecast", "town"]


for year in sorted(glob(PATH)):
    if "Window" in platform.platform():
        _year = year.split(seperator)[-1]
    else:
        _year = unicodedata.normalize('NFC', year.split(seperator)[-1])

    for city in sorted(glob(year + "/*")):
        if "Window" in platform.platform():
            _city = city.split(seperator)[-1]
        else:
            _city = unicodedata.normalize('NFC', city.split(seperator)[-1])

        for kind in measurement.keys():
            content = {col: [] for col in COLUMNS}
            content[measurement[kind]] = []

            # unicode test
            # temp1 = ''.join(r'\u{:04X}'.format(ord(chr)) for chr in "강수형태")
            # temp2 = ''.join(r'\u{:04X}'.format(ord(chr)) for chr in "강수형태")

            if "Window" in platform.platform():
                files = glob(city + "/*[" + kind + "]*.csv")    # regex
            else:
                # 추후 re 사용해 regex 정식으로 테스트
                f_decode = [unicodedata.normalize('NFC', f_uni) for f_uni in glob(city + "/*.csv")]
                files = [f_encode for f_encode in f_decode if kind in f_encode]

            for file in tqdm(sorted(files)):
                file_name_parse = file.split("/")[-1].split("_")
                town = file_name_parse[0].split(seperator)[-1]
                start_date = file_name_parse[2]

                with open(file, encoding="utf-8") as f:
                    f.readline()  # remove first line
                    for i, line in enumerate(f.readlines(), start=1):
                        if "Start" in line:
                            start_date = line.replace(" ", "").split("Start:")[1].split(",")[0][:6]
                            memo = line     # to check date format error
                        elif not ((len(line) < 1) or ("NA" in line) or ("format..day,hour,forecast,value" in line)):
                            if "\n" in line:
                                line = line.replace("\n", "")

                            # 다음과 같은 경우 예방 => ['" 1"', '"200"', '"4"', '281']
                            data = line.replace("+", "").replace("\"", "").split(",")

                            if (len(data) > 3) and (data.count("") < 1):
                                data[0] = data[0].replace(" ", "")
                                data[1] = data[1].replace(" ", "")
                                temp_date = start_date + \
                                            ("0" + data[0] if int(data[0]) < 10 else data[0]) + \
                                            ("0" + data[1] if len(data[1]) < 4 else data[1])

                                # raw data에는 영국 시간 기준이므로 8시간을 더해 한국 시간으로 변경
                                # 따라서 데이터도 다음달로 밀리는 경우 존재(e.g. 2012-12-31 22:00 => 2013-01-01 06:00)
                                date = datetime.strptime(temp_date, "%Y%m%d%H%M") + timedelta(hours=8)
                                content["date"].append(date)
                                content["year"].append(date.year)
                                content["month"].append(date.month)
                                content["day"].append(date.day)
                                content["hour"].append(date.hour)
                                content["forecast"].append(int(data[2]))
                                content["town"].append(unicodedata.normalize('NFC', town))
                                content[measurement[kind]].append(float(data[3]))

            if len(content) > 0:
                df = pd.DataFrame(content)
                os.makedirs("result/" + kind, exist_ok=True)
                df.to_csv("result/" + kind + "/" + _year + "_" + _city + "_" + kind + ".csv",
                          index=False, encoding="utf-8-sig")
                print("\n================================ " + _year + " " + _city + " " + kind + " is completed ================================\n")

print("Finish merge Successfully")
