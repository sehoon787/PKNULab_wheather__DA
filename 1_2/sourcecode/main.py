import TemperatureModel as tm


if __name__ == '__main__':
    # fm = tm.FileMaker(dir_path="", res_path="")
    # fm.mergeFile(load_path="", target_path="과제1-2_결측제거_0730_1", fileName="only78", resampling=False)

    # train data
    fm = tm.FileMaker(dir_path="과제2 데이터", res_path="과제1-2_결측제거_0731_1")
    df, outlier_info_df, res_path = tm.IQR(path=fm.run()).run(remove_col=[], save_plot=False, draw_plot=False)

    # test data
    # answer_fm = tm.FileMaker(dir_path="과제2 데이터", res_path="과제1-2_결측제거_0730_1")
    # tm.TestDataMaker(path=answer_fm.run())

    ## test data 0731
    # import pandas
    # tm2 = tm.TestDataMaker(path="20201105.LST.csv")
    # tm2.run(df=pandas.read_csv("20201105.LST.csv"), res_path="iqr")

