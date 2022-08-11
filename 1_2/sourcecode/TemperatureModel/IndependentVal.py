from __init__ import extract_col

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def get_corrleation(df, res_path):
    print(df.describe())
    df.describe()

    corr_result_df = df[extract_col].corr()
    corr_result_df.to_csv(res_path + "/corr_result.csv")

    return corr_result_df


def get_vif(x_train):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(x_train.values, i)
                         for i in range(x_train.shape[1])]
    vif['Feature'] = x_train.columns
    return vif


def vif_ols(df, target='isitu-LST'):
    df = df[extract_col]
    print(df.shape)

    vif = get_vif(df.drop(columns=[target], axis=1))
    print(vif)

    df['intercept'] = 1
    model = sm.OLS(df[target], df.drop(columns=[target], axis=1))
    results = model.fit()

    print(results.summary())

    return model, results
