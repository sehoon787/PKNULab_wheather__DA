from datetime import datetime
import numpy as np
import pandas as pd


def drop_na():
    df = pd.read_csv("result3.csv", encoding="utf-8-sig")
    x_2013 = df.loc[df['year'] == 2013].dropna(axis=1)

    df2 = df[x_2013.keys()]

    df3 = df2.loc[df2['year'] == 2016].dropna(axis=1)

    return df[df3.keys()].dropna(axis=0)


def load_train_bh(path):
    df = pd.read_csv(path, encoding="cp949")
    df.columns = ["date", "city", "gender", "frequency"]

    year = []
    month = []
    day = []
    for i in range(len(df)):
        date = datetime.strptime(str(df["date"].iloc[i]), "%Y%m%d")
        year.append(date.year)
        month.append(date.month)
        day.append(date.day)

    df["year"] = year
    df["month"] = month
    df["day"] = day

    return df[["city", "year", "month", "day", "gender", "frequency"]]


def load_test_bh(path):
    df = pd.read_csv(path, encoding="cp949")
    df.columns = ["date", "city", "gender", "frequency"]

    year = []
    month = []
    day = []
    for i in range(len(df)):
        date = datetime.strptime(str(df["date"].iloc[i]), "%Y-%m-%d")
        year.append(date.year)
        month.append(date.month)
        day.append(date.day)

    df["year"] = year
    df["month"] = month
    df["day"] = day

    return df[["city", "year", "month", "day", "gender", "frequency"]]


x = drop_na()
bh1 = load_train_bh("data/bh.csv")
res1 = pd.merge(x, bh1, how="right").dropna(axis=0)
res1[res1.keys()[80:-2]] = res1[res1.keys()[80:-2]]/10000
res1.to_csv("result4_train.csv", index=False, encoding="utf-8-sig")


bh2 = load_test_bh("data/2-2_검증데이터셋.csv")
res2 = pd.merge(x, bh2, how="right")
res2[res1.keys()[80:-2]] = res2[res2.keys()[80:-2]]/10000
res2.to_csv("result4_test.csv", index=False, encoding="utf-8-sig")




