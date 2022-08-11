# -*- coding: utf-8 -*-
from __init__ import *

from glob import glob
import numpy as np
import os
import pandas as pd
import platform
from tqdm import tqdm
import unicodedata


seperator = "\\" if "Window" in platform.platform() else "/"

PATH = "result/*"
COLUMNS = ["date", "year", "month", "day", "hour", "forecast", "town"]

for kind in sorted(glob(PATH)):
    content = []

    if "Window" in platform.platform():
        _kind = kind.split(seperator)[-1]
    else:
        _kind = unicodedata.normalize('NFC', kind.split(seperator)[-1])

    for i, city in tqdm(enumerate(cities)):
        if "Window" in platform.platform():
            files = glob(kind + "/*[" + city + "]*.csv")  # regex
        else:
            f_decode = [unicodedata.normalize('NFC', f_uni) for f_uni in glob(kind + "/*.csv")]
            files = [f_encode for f_encode in f_decode if city in f_encode]

        for j, file in enumerate(sorted(files)):
            with open(file, encoding="utf-8") as f:
                columns = f.readline()  # remove first line
                if i == 0:
                    content.append((unicodedata.normalize('NFC', columns).replace("\n", "")+",city").split(","))
                for k, line in enumerate(f.readlines()):
                    if "\n" in line:
                        line = line.replace("\n", "")
                    if len(line) > 0:
                        content.append((line+","+city).split(","))

    df = pd.DataFrame(content[1:], columns=content[0])[['year', 'month', 'day', 'forecast', "city", measurement[_kind]]]
    sample_df = df.drop(df.loc[df[measurement[_kind]] == measurement[_kind]].index)
    sample_df[measurement[_kind]] = sample_df[measurement[_kind]].astype(np.float32)
    result_df = sample_df.groupby(['year', 'month', 'day', 'forecast', "city"], as_index=False).mean()

    os.makedirs("result2/", exist_ok=True)
    result_df.to_csv("result2/" + _kind + ".csv", index=False, encoding="utf-8-sig")

    print("\n================================ " + _kind + " is completed ================================\n")

print("Finish merge Successfully")





