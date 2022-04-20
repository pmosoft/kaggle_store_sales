from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
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
# pliot = True
pliot = False
###############################################################################
#                                      구현부
###############################################################################

###########################################################
# pliot
###########################################################
if pliot:
    # %%
    _trans = load_data.train_master('train_master_all')

    start_dtm = dt.get_now()
    vo = experiment_vo()
    vo.scenario_id = 'x001d001y001m002'
    vo.scenario_desc = '기본 feature Ridge 모델'
    vo.feature_col = 'date8, month2, day2, onpromotion, transactions'
    vo.feature_sdt8 = '20170101'
    vo.feature_edt8 = '20170730'
    vo.predict_col = 'date8, sales'
    vo.predict_sdt8 = '20170801'
    vo.predict_edt8 = '20170815'
    vo.model_name = 'Ridge'
    vo.store_nbr = 1
    vo.family2 = 4
    train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=31.0))

    train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=31.0))
    model = ridge.fit(train_X, train_y)
    predict_y = pd.DataFrame(model.predict(test_X), index=test_X.index, columns=test_y.columns).clip(0.0)
    vo.mse = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
    vo.score = float("{:.2f}".format(1 - mean_squared_log_error(test_y, predict_y)))

    score = vo.score
    # plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_y['date8'], test_y['sales'], predict_y['sales'])


###########################################################
# 모델 fit, predict
###########################################################

def execute(scenario_id, scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_sdt8, predict_edt8, model_name, model_cfg):
    _trans = load_data.train_master(tab_nm)

    _vo_list = []
    work_dtm16 = dt.get_time_str16();
    work_dtm15 = dt.get_time_str15()

    store_family_df = make_train_test_dataset.store_family_df(_trans)

    # for store_nbr in range(54):
    #     for family2 in range(33):
    for i in store_family_df.index:
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
        vo.predict_col = 'sales'
        vo.predict_sdt8 = predict_sdt8
        vo.predict_edt8 = predict_edt8
        vo.model_name = model_name
        vo.model_cfg = model_cfg
        vo.store_nbr = store_nbr
        vo.family2 = family2

        train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)
        ridge = make_pipeline(RobustScaler(), Ridge(alpha=31.0))
        model = ridge.fit(train_X, train_y)
        predict_y = pd.DataFrame(model.predict(test_X), index=test_X.index, columns=test_y.columns).clip(0.0)

        vo.mse = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
        vo.score = float("{:.2f}".format(1 - mean_squared_log_error(test_y, predict_y)))
        vo.test_y = test_y['sales'].tolist()
        vo.predict_y = predict_y['sales'].tolist()
        vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())
        _vo_list.append(vo.__dict__)
        print(f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
        # plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_y['date8'], test_y['sales'], predict_y['sales'])
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list


if __name__ == '__main__':
    print(">>>>> main")

    execute('x001f001d001y001m002c001', '전체건-기본 feature1-전기간-Ridge 모델', 'train_master_all', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x001f001d002y001m002c001', '전체건-기본 feature1-전기간-Ridge 모델', 'train_master_all', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x001f002d001y001m002c001', '전체건-기본 feature2-특정기간-Ridge 모델', 'train_master_all', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x001f002d002y001m002c001', '전체건-기본 feature2-특정기간-Ridge 모델', 'train_master_all', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')

    execute('x002f001d001y001m002c001', 'sales 존재-기본 feature1-전기간-Ridge 모델', 'train_master_exist_sales', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x002f001d002y001m002c001', 'sales 존재-기본 feature1-전기간-Ridge 모델', 'train_master_exist_sales', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x002f002d001y001m002c001', 'sales 존재-기본 feature2-특정기간-Ridge 모델', 'train_master_exist_sales', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x002f002d002y001m002c001', 'sales 존재-기본 feature2-특정기간-Ridge 모델', 'train_master_exist_sales', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')

    execute('x003f001d001y001m002c001', '상점별 기간-기본 feature1-전기간-Ridge 모델', 'train_master_store_dates', 'date8, month2, day2, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x003f001d002y001m002c001', '상점별 기간-기본 feature1- 전기간-Ridge 모델', 'train_master_store_dates', 'date8, month2, day2, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x003f002d001y001m002c001', '상점별 기간-기본 feature2-특정기간-Ridge 모델', 'train_master_store_dates', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170730', '20170801', '20170815', 'LinearRegression', '')
    execute('x003f002d002y001m002c001', '상점별 기간-기본 feature2-특정기간-Ridge 모델', 'train_master_store_dates', 'date8, month2, day2, day_of_week, onpromotion, transactions', '20170101', '20170730', '20170801', '20170815', 'LinearRegression', '')

    df_qry = predict_analysis.scenario_score_rate()
