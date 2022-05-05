from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

import numpy as np
import pandas as pd
import json

from feature import load_data
from feature import make_train_test_dataset
from model.experiment_vo import experiment_vo
import model.scenario_id_code as scd

import util.date_util as dt
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

# %%
###########################################################
# pliot
###########################################################
if pliot:

    start_dtm = dt.get_now()
    scenario_id = 'x003f001d001y001t001m001c001'
    scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_col, predict_sdt8, predict_edt8, model_name = scd.get_code_name(scenario_id)
    _trans = load_data.train_master(tab_nm)
    # %%
    vo = experiment_vo()
    vo.scenario_id = scenario_id
    vo.feature_col = feature_col
    vo.feature_sdt8 = feature_sdt8
    vo.feature_edt8 = feature_edt8
    vo.predict_col = predict_col
    vo.predict_sdt8 = predict_sdt8
    vo.predict_edt8 = predict_edt8
    vo.model_name = model_name
    vo.store_nbr = 1
    vo.family2 = 1
    train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)

    model = LinearRegression().fit(train_X, train_y)
    predict_y = pd.DataFrame(model.predict(test_X).astype(int), index=test_X.index, columns=test_y.columns).clip(0.0)

    vo.mae = float("{:.2f}".format(mean_absolute_error(test_y, predict_y)))
    vo.mse = float("{:.2f}".format(mean_squared_error(test_y, predict_y)))
    vo.rmse = float("{:.2f}".format(np.sqrt(mean_squared_error(test_y, predict_y))))
    vo.msle = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
    vo.rmsle = float("{:.2f}".format(np.sqrt(mean_squared_log_error(test_y, predict_y))))
    vo.r2 = float("{:.2f}".format(r2_score(test_y, predict_y)))
    vo.score = float("{:.2f}".format(1 - vo.msle))
    vo.test_y = test_y['sales'].tolist()
    vo.predict_y = predict_y['sales'].tolist()

    score = vo.score
    # plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_X['date8'], test_y['sales'], predict_y['sales'])


###########################################################
# 모델 fit, predict
###########################################################

def execute(scenario_id, model_cfg):
    scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_col, predict_sdt8, predict_edt8, model_name = scd.get_code_name(scenario_id)

    _trans = load_data.train_master(tab_nm)

    _vo_list = []
    work_dtm16 = dt.get_time_str16()
    work_dtm15 = dt.get_time_str15()
    # for store_nbr in range(54):
    #     for family2 in range(33):
    store_family_df = make_train_test_dataset.store_family_df(_trans)

    # for store_nbr in range(54):
    #     for family2 in range(33):
    for i in store_family_df.index:
    #for i in range(1):
        store_nbr = int(store_family_df['store_nbr'][i])
        family2 = int(store_family_df['family2'][i])
        start_dtm = dt.get_now()

        vo = experiment_vo()
        vo.work_dtm16 = work_dtm16
        vo.scenario_id = scenario_id
        vo.scenario_desc = scenario_desc
        vo.feature_src = tab_nm
        vo.feature_col = feature_col
        vo.feature_sdt8 = feature_sdt8
        vo.feature_edt8 = feature_edt8
        vo.predict_col = predict_col
        vo.predict_sdt8 = predict_sdt8
        vo.predict_edt8 = predict_edt8
        vo.model_name = model_name
        vo.model_cfg = model_cfg
        vo.store_nbr = store_nbr
        vo.family2 = family2

        train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)

        model = LinearRegression().fit(train_X, train_y)
        predict_y = pd.DataFrame(model.predict(test_X).astype(int), index=test_X.index, columns=test_y.columns).clip(0.0)
        vo.mae = float("{:.2f}".format(mean_absolute_error(test_y, predict_y)))
        vo.mse = float("{:.2f}".format(mean_squared_error(test_y, predict_y)))
        vo.rmse = float("{:.2f}".format(np.sqrt(mean_squared_error(test_y, predict_y))))
        vo.msle = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
        vo.rmsle = float("{:.2f}".format(np.sqrt(mean_squared_log_error(test_y, predict_y))))
        vo.r2 = float("{:.2f}".format(r2_score(test_y, predict_y)))
        vo.score = float("{:.2f}".format(1 - vo.msle))

        vo.test_y = test_y['sales'].tolist()
        vo.predict_y = predict_y['sales'].tolist()

        vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())
        _vo_list.append(vo.__dict__)
        print(f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list


if __name__ == '__main__':
    print(">>>> main")
    # scd.get_code_name('x001f005d001y001t001m001c001')

    # #################################
    # # Evaluate
    # #################################
    #execute('x001f006d002y001t001m001c001', '')

    # execute('x001f005d001y001t001m001c001', '')
    # execute('x001f005d002y001t001m001c001', '')
    #
    # execute('x002f005d001y001t001m001c001', '')
    # execute('x002f005d002y001t001m001c001', '')

    execute('x003f005d001y001t001m001c001', '')
    execute('x003f005d002y001t001m001c001', '')

    # #################################
    # # Predict
    # #################################
    execute('x001f005d003y001t002m001c001', '')
    execute('x001f005d004y001t002m001c001', '')

    execute('x002f005d003y001t002m001c001', '')
    execute('x002f005d004y001t002m001c001', '')

    execute('x003f005d003y001t002m001c001', '')
    execute('x003f005d004y001t002m001c001', '')

    # execute('x001f004d001y001m001c001', '전체건-기본 feature2-특정기간-회귀 모델', 'train_master_all', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')

    # execute('x001f001d001y001m001c001', '전체건-기본 feature1-전기간-회귀 모델', 'train_master_all', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x001f001d002y001m001c001', '전체건-기본 feature1-전기간-회귀 모델', 'train_master_all', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x001f002d001y001m001c001', '전체건-기본 feature2-특정기간-회귀 모델', 'train_master_all', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x001f002d002y001m001c001', '전체건-기본 feature2-특정기간-회귀 모델', 'train_master_all', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    #
    # execute('x002f001d001y001m001c001', 'sales 존재-기본 feature1-전기간-회귀 모델', 'train_master_exist_sales', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x002f001d002y001m001c001', 'sales 존재-기본 feature1-전기간-회귀 모델', 'train_master_exist_sales', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x002f002d001y001m001c001', 'sales 존재-기본 feature2-특정기간-회귀 모델', 'train_master_exist_sales', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x002f002d002y001m001c001', 'sales 존재-기본 feature2-특정기간-회귀 모델', 'train_master_exist_sales', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    #
    # execute('x003f001d001y001m001c001', '상점별 기간-기본 feature1-전기간-회귀 모델', 'train_master_store_dates', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x003f001d002y001m001c001', '상점별 기간-기본 feature1- 전기간-회귀 모델', 'train_master_store_dates', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x003f002d001y001m001c001', '상점별 기간-기본 feature2-특정기간-회귀 모델', 'train_master_store_dates', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    # execute('x003f002d002y001m001c001', '상점별 기간-기본 feature2-특정기간-회귀 모델', 'train_master_store_dates', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')

    #df_qry = predict_analysis.scenario_score_rate()
