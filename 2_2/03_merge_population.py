# -*- coding: utf-8 -*-
from __init__ import *
from glob import glob
import pandas as pd
from tqdm import tqdm
import unicodedata
import re

PATH = "result2/*"


def make_col_names(df):
    col = []
    for k in df.keys():
        _isFound = False
        for i in range(len(cities)):
            if re.match("^" + cities[i][0] + "[가-힣]*" + cities[i][1] + "[가-힣]", k):
                _isFound = True
                col.append(cities[i])
                break
        if not _isFound:
            col.append(k)

    return col


def make_gender_population_data():
    df = pd.read_csv("/Users/sehunkim/Desktop/proj/공모전/과제2-2/sourcecode/data/인구/남녀성비_시도_시_군_구__20220805020359.csv",
                     encoding="cp949")
    df = df.set_index([df.keys()[0]]).transpose()
    df.columns = make_col_names(df)
    year = []
    month = []

    for d in df.index:
        date = d.split(".")
        if len(date) > 1:
            year.append(int(date[0]))
            month.append(int(date[1]))
        else:
            year.append(0)
            month.append(0)

    df["행정구역별(1)"] = [gen.split("<br>")[0] for gen in df["행정구역별(1)"]]

    sample_df = df.set_index([df.keys()[0]])[cities]
    sample_df["year"] = year
    sample_df["month"] = month

    return sample_df.reset_index(level=["행정구역별(1)"])


def make_age_population_data():
    merge_df = pd.DataFrame()
    for file in tqdm(sorted(glob("/Users/sehunkim/Desktop/proj/공모전/과제2-2/sourcecode/data/인구/age_pop/*"))):
        content1 = []
        content2 = []
        content3 = []

        with open(file, encoding="cp949") as f:
            a_date = file.split("/")[-1].split("_")[0]
            year = a_date[:4]
            month1 = int(a_date[4:])
            month2 = month1+1
            month3 = month2+1

            for i, line in enumerate(f.readlines()):
                if i == 0:
                    modified_col = []
                    for cc in line.replace("\n", "").replace("\"", "").split(","):
                        if ("년" in cc) and ("월" in cc) and ("남" in cc):
                            modified_col.append("man_" + cc.split("남_")[-1].replace("세", ""))
                        elif ("년" in cc) and ("월" in cc) and ("여" in cc):
                            modified_col.append("woman_" + cc.split("여_")[-1].replace("세", ""))

                    col = modified_col[:int(len(modified_col)/3)]
                    col.insert(0, "행정구역")
                    col.append("year")
                    col.append("month")

                    content1.append(col)
                    content2.append(col)
                    content3.append(col)

                else:
                    data = line.replace(",", "").replace("\n", "").split("\"\"")
                    data[0] = data[0].replace("\"", "")
                    data[-1] = data[-1].replace("\"", "")

                    data1 = data[:int(len(data)/3)+1][24:]
                    data2 = data[int(len(data)/3):int(len(data)/3*2)+1][24:]
                    data3 = data[int(len(data)/3*2):][24:]

                    data1.insert(0, data[0])
                    data2.insert(0, data[0])
                    data3.insert(0, data[0])

                    data1.append(year)
                    data1.append(month1)
                    data2.append(year)
                    data2.append(month2)
                    data3.append(year)
                    data3.append(month3)

                    content1.append(data1)
                    content2.append(data2)
                    content3.append(data3)

        df1 = pd.DataFrame(content1[1:], columns=content1[0])
        df2 = pd.DataFrame(content2[1:], columns=content2[0])
        df3 = pd.DataFrame(content3[1:], columns=content3[0])
        df = pd.concat([df1, df2, df3], axis=0)

        merge_df = pd.concat([merge_df, df], axis=0)

    merge_df = merge_df.transpose()
    remove_idx = [idx for idx in merge_df.index if (("총인구수" in idx) or ("연령구간인구수" in idx))]
    merge_df = merge_df.drop(remove_idx)
    merge_df = merge_df.transpose()

    city = [province for province in merge_df["행정구역"] if len(province.split("  ")) == 2]
    sample_df = merge_df.set_index([merge_df.keys()[0]]).transpose()[city]
    sample_df.columns = make_col_names(sample_df)
    sample_df = sample_df[cities].transpose()
    sample_df.index.name = "city"
    sample_df = sample_df.reset_index(level=["city"])

    sample_df = sample_df.drop_duplicates(keep='first')

    return sample_df


def merge_weather_data():
    result = pd.DataFrame([])

    for file in tqdm(sorted(glob(PATH))):
        for m in measurement.keys():
            if m in unicodedata.normalize('NFC', file):
                _kind = m
                break
        sample_df = pd.read_csv(file, encoding="utf-8-sig")

        forecast = {k: [] for k in set(sample_df["forecast"])}
        for i, data in enumerate(sample_df[measurement[_kind]]):
            forecast[sample_df["forecast"].iloc[i]].append(
                [sample_df["year"].iloc[i], sample_df["month"].iloc[i], sample_df["day"].iloc[i],
                 sample_df["city"].iloc[i], data])

        for k in forecast:
            df1 = pd.DataFrame(forecast[k], columns=["year", "month", "day", "city", _kind + "_" + str(k)])
            df1 = df1.set_index([df1.keys()[0], df1.keys()[1], df1.keys()[2], df1.keys()[3]])
            result = pd.concat([result, df1], axis=1)

    return result.reset_index(level=["year", "month", "day", "city"])


def merge_final_data():
    weather_df = merge_weather_data()
    weather_df = weather_df.drop(weather_df[weather_df['year'] > 2016].index)

    gen_population_df = make_gender_population_data()
    age_population_df = make_age_population_data()

    gen_data_kind = ["남녀성비", "남자인구수", "여자인구수"]
    pop_ratio = []
    man_pop = []
    women_pop = []

    for i, city in tqdm(enumerate(weather_df['city'])):
        gen_row1 = gen_population_df[city].loc[
            (gen_population_df["행정구역별(1)"] == gen_data_kind[0]) &
            (gen_population_df['year'] == weather_df['year'].iloc[i]) &
            (gen_population_df['month'] == weather_df['month'].iloc[i])
            ]
        gen_row2 = gen_population_df[city].loc[
            (gen_population_df["행정구역별(1)"] == gen_data_kind[1]) &
            (gen_population_df['year'] == weather_df['year'].iloc[i]) &
            (gen_population_df['month'] == weather_df['month'].iloc[i])
            ]
        gen_row3 = gen_population_df[city].loc[
            (gen_population_df["행정구역별(1)"] == gen_data_kind[2]) &
            (gen_population_df['year'] == weather_df['year'].iloc[i]) &
            (gen_population_df['month'] == weather_df['month'].iloc[i])
            ]

        pop_ratio.append(gen_population_df[city][gen_row1.index[0]])
        man_pop.append(gen_population_df[city][gen_row2.index[0]])
        women_pop.append(gen_population_df[city][gen_row3.index[0]])

    weather_df["pop_ratio"] = pop_ratio
    weather_df["man_pop"] = man_pop
    weather_df["women_pop"] = women_pop
    weather_df["year"] = weather_df["year"].astype(int)
    weather_df["month"] = weather_df["month"].astype(int)
    weather_df = weather_df.sort_values(by=["year", "month"], ascending=True)

    age_population_df["year"] = age_population_df["year"].astype(int)
    age_population_df["month"] = age_population_df["month"].astype(int)
    age_population_df = age_population_df.sort_values(by=["year", "month"], ascending=True)

    return pd.merge(weather_df, age_population_df, how="right")


res = merge_final_data()
res.to_csv("result3.csv", index=False, encoding="utf-8-sig")
