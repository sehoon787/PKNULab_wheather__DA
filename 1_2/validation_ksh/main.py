import TemperatureModel as tm

import pandas as pd
if __name__ == '__main__':
    fm = tm.FileMaker(dir_path="", res_path="")
    fm.mergeFile(load_path="/Users/sehunkim/Desktop/proj/wheather/data/과제2 결측제거",
                       target_path="/Users/sehunkim/Desktop/proj/wheather/validation_ksh/resample", remove_month=[1, 2, 3, 4, 5, 9, 10, 11, 12],
                       fileName="merge_data_only789", resampling=True)

    # train data
    # fm = tm.FileMaker(dir_path="과제2 데이터", res_path="과제1-2_결측제거_0730_1")
    # df, outlier_info_df, res_path = tm.IQR(path=fm.run()).run(remove_col=[], save_plot=False, draw_plot=False)

    # test data
    # answer_fm = tm.FileMaker(dir_path="과제2 데이터", res_path="과제1-2_결측제거_0730_1")
    # tm.TestDataMaker(path=answer_fm.run())

