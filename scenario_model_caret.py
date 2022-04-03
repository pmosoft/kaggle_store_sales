''' setup 이후 prompt 상태로 떨어저 auto execute 불가로 실험 중단 '''
from pycaret.regression import *
from sklearn.metrics import mean_squared_log_error

import pandas as pd
import json

from feature import load_data
from feature import make_train_test_dataset
from util.vo import experiment_vo
import util.date_util as dt
import util.plot_util as plot_util
import scenario_predict_analysis as predict_analysis

###############################################################################
#                                    함수 정의
###############################################################################
'''
pilot
execute
analysis
visual 
main
'''
###############################################################################
#                                     전역변수
###############################################################################
vo_list = []
path = "d:/lge/pycharm-projects/kaggle_store_sales/output/"
#pliot = True
pliot = False
###############################################################################
#                                      구현부
###############################################################################

###########################################################
# pliot
###########################################################
# %%
#_trans = load_data.train_master2()

# %%
if pliot :
    _trans = load_data.train_master2()

    start_dtm = dt.get_now()

    vo = experiment_vo()
    vo.scenario_id = 'x001d002y001m001'
    vo.scenario_desc = '기본 feature autoML pycaret 모델'
    vo.feature_col = 'date8, month2, day2, onpromotion, transactions'
    vo.feature_sdt8 = '20170101'
    vo.feature_edt8 = '20170730'
    vo.predict_col = 'sales'
    vo.predict_sdt8 = '20170801'
    vo.predict_edt8 = '20170815'
    vo.model_name = 'pycaret'
    vo.store_nbr = 1
    vo.family2 = 4

    train, test = make_train_test_dataset.query2(_trans, vo)

    setup(data=train, target='sales')
    best = compare_models()
    predict_y = predict_model(best, data=test)['Label'].clip(0.0)

    vo.mse = float("{:.2f}".format(mean_squared_log_error(test['sales'], predict_y)))
    vo.score = float("{:.2f}".format(1 - vo.mse))
    vo.test_y = test['sales'].tolist()
    vo.predict_y = predict_y.tolist()
    vo.auto_ml_model = str(best)
    vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())

    #plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_X['date8'], test_y['sales'], predict_y['sales'])
#%%
#predict_model(best)
#predictions = predict_model(best, data=test)
#predict_y2 = predict_y.tolist()
#predictions.head()
#get_model_name(best())
#print(str(best))
#aaa = str(best)
#vo.auto_ml_model = str(best)

#%%

###########################################################
# 모델 fit, predict
###########################################################

def execute(trans,scenario_id,feature_col,feature_sdt8,feature_edt8,predict_sdt8,predict_edt8):
    _vo_list = []
    work_dtm16 = dt.get_time_str16(); work_dtm15 = dt.get_time_str15()
    # for store_nbr in range(54):
    #     for family2 in range(33):
    for store_nbr in range(54):
        for family2 in range(33):
            start_dtm = dt.get_now()

            vo = experiment_vo()
            vo.work_dtm16    = work_dtm16
            vo.scenario_id   = scenario_id
            vo.scenario_desc = '기본 feature 기본 모델 회귀 only'
            vo.feature_col   = feature_col
            vo.feature_sdt8  = feature_sdt8
            vo.feature_edt8  = feature_edt8
            vo.predict_col   = 'sales'
            vo.predict_sdt8  = predict_sdt8
            vo.predict_edt8  = predict_edt8
            vo.model_name = 'pycaret'
            vo.store_nbr = store_nbr+1
            vo.family2 = family2 + 1

            train, test = make_train_test_dataset.query2(_trans, vo)

            setup(data=train, target='sales')
            best = compare_models()
            predict_y = predict_model(best, data=test)['Label'].clip(0.0)

            vo.mse = float("{:.2f}".format(mean_squared_log_error(test['sales'], predict_y)))
            vo.score = float("{:.2f}".format(1 - vo.mse))
            vo.test_y = test['sales'].tolist()
            vo.predict_y = predict_y.tolist()
            vo.auto_ml_model = str(best)
            vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())

            _vo_list.append(vo.__dict__)
            print(f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list

if __name__ == '__main__':
    print(">>>> main")
    _trans = load_data.train_master2()
    execute(_trans,'x002d002y001m004','date8, month2, day2, day_of_week, onpromotion, transactions','20170101','20170730','20170801','20170815')
    df_qry = predict_analysis.scenario_score_rate()
