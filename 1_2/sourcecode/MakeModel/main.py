from train_Data import my_train
from validation_Data import validation

if __name__ == '__main__':
    # train data
    # lstm 계층 구조는 lstm.py에서 수정
    my_lst_model = my_train(train_data="../train/원본_학습용_56789_0802.csv",
                            train_type="lst",
                            model_path="../model",
                            model_name="pknu_lstm5",
                            epoch=100,
                            batch_size=1024)


    #validation file : res_ref(제출, report), vali_211001(11월로 검증, check)
    validation(validation_file = "../validation/res_ref.csv",
                  validation_type="report",
                  # model_lst=my_lst_model,
                  model_lst="../model/pknu_lstm4.h5",
                  model_ta="../model/pknu_lstm4_56789_TA_2.h5",
                  result_name="my_validation.csv")