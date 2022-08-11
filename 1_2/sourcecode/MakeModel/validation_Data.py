from __init__ import colnames

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import tensorflow as tf


def validation(validation_file="../validation/res_ref.csv",
               validation_type="check",
               model_lst="../model/train_model_lst.h5",
               model_ta="../model/train_model_ta.h5",
               result_name="my_validation.csv") :
    '''
    :param validation_file: load validation file path (defalut : "../validation/res_ref.csv")
    :param validation_type: check / report (defalut : "check")
    :param model_lst: lst model path (defalut : "../model/train_model_lst.h5")
    :param model_ta: ta model path (defalut : "../model/train_model_ta.h5")
    :param result_name: predict data csv file (defalut : "my_validation.csv")
    '''

    test_df = pd.read_csv(validation_file)


    if validation_type == "check" :
        # Attach month column
        test_df['month'] = pd.to_datetime(test_df['YearMonthDayHourMinute'], format='%Y%m%d%H%M').dt.strftime('%m').astype(int)

        # Attach time column
        test_df['time'] = pd.to_datetime(test_df['YearMonthDayHourMinute'], format='%Y%m%d%H%M').dt.strftime('%H%M').astype(int)

        test_df['LandType'] += 1         # => 기존 LandType의 카테고리에서 1을 더함. 0에 가중치가 곱해지는 것을 방지하기 위함

        test_df = test_df.replace(0, 0.00001)

        test_df = test_df[colnames]

        for i in colnames:
            test_df = test_df[(test_df[i] != -999)]


    X = test_df.loc[:, colnames[2:]].values

    model_lst_load = tf.keras.models.load_model(model_lst)
    model_ta_load = tf.keras.models.load_model(model_ta)

    y_score_lst = model_lst_load.predict(X)
    y_score_ta = model_ta_load.predict(X)

    if validation_type == "check" :
        y_lst = test_df.loc[:, colnames[0]].values
        y_ta = test_df.loc[:, colnames[1]].values

        print("y_score_lst")
        print("MSE: " + str(mean_squared_error(y_lst, y_score_lst)))
        print("RMSE: " + str(np.sqrt(mean_squared_error(y_lst, y_score_lst))))
        print("MAE: " + str(mean_absolute_error(y_lst, y_score_lst)))

        print("y_score_ta")
        print("MSE: " + str(mean_squared_error(y_ta, y_score_ta)))
        print("RMSE: " + str(np.sqrt(mean_squared_error(y_ta, y_score_ta))))
        print("MAE: " + str(mean_absolute_error(y_ta, y_score_ta)))



    df_lst = pd.DataFrame(y_score_lst)
    df_ta = pd.DataFrame(y_score_ta)

    if validation_type == "report" :
        pd.concat([df_lst, df_ta], axis=1, ignore_index=True).to_csv(result_name, encoding="utf-8", index=False, header=['lst', 'ta'])

        df_report = pd.read_csv("../validation/1-2_검증데이터셋_원본.csv")

        df_report["isitu-LST"] = df_lst
        df_report["insitu-TA"] = df_ta

        df_report.to_csv("220241.csv", encoding="utf-8", index=False)

    elif validation_type == "check" :
        answer_lst = pd.DataFrame(y_lst)
        answer_ta = pd.DataFrame(y_ta)

        pd.concat([df_lst, answer_lst, df_ta, answer_ta], axis=1, ignore_index=True).to_csv(result_name, encoding="utf-8", index=False, header=['lst', 'answer_lst', 'ta', 'answer_ta'])
