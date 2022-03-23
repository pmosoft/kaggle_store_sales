from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor
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
#pliot = True
pliot = False
###############################################################################
#                                      구현부
###############################################################################

###########################################################
# pliot
###########################################################
if pliot :
    # %%
    _trans = load_data.train_master2()

    vo = experiment_vo()
    vo.scenario_id = 'x001d002y001m003'
    vo.scenario_desc = '기본 feature 복합 모델'
    vo.feature_col = 'date8, month2, day2, onpromotion, transactions'
    vo.feature_sdt8 = '20170101'
    vo.feature_edt8 = '20170730'
    vo.predict_col = 'sales'
    vo.predict_sdt8 = '20170801'
    vo.predict_edt8 = '20170815'
    vo.model_name = 'CustomRegression'
    vo.store_nbr = 1
    vo.family2 = 4
    train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)

    n_jobs = -1; seed = 1
    if vo.family2 == 32: # SCHOOL AND OFFICE SUPPLIES
        etr = ExtraTreesRegressor(n_estimators=500, n_jobs=n_jobs, random_state=seed)
        rfr = RandomForestRegressor(n_estimators=500, n_jobs=n_jobs, random_state=seed)
        br1 = BaggingRegressor(base_estimator=etr, n_estimators=10, n_jobs=n_jobs, random_state=seed)
        br2 = BaggingRegressor(base_estimator=rfr, n_estimators=10, n_jobs=n_jobs, random_state=seed)
        model = VotingRegressor([('ExtraTrees', br1), ('RandomForest', br2)])
    else:
        ridge = make_pipeline(RobustScaler(), Ridge(alpha=31.0, random_state=seed))
        svr = make_pipeline(RobustScaler(), SVR(C=1.68, epsilon=0.09, gamma=0.07))
        model = VotingRegressor([('ridge', ridge), ('svr', svr)])

    aaa = train_y.values.ravel()
    model.fit(train_X, train_y.values.ravel())
    predict_y = pd.DataFrame(model.predict(test_X), index=test_X.index, columns=test_y.columns).clip(0.0)
    vo.mse = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
    vo.score = float("{:.2f}".format(1 - mean_squared_log_error(test_y, predict_y)))
    vo.test_y = test_y['sales'].tolist()
    vo.predict_y = predict_y['sales'].tolist()

    plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_X['date8'], test_y['sales'], predict_y['sales'])


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
            vo.scenario_desc = '기본 feature 복합 모델'
            vo.feature_col   = feature_col
            vo.feature_sdt8  = feature_sdt8
            vo.feature_edt8  = feature_edt8
            vo.predict_col   = 'sales'
            vo.predict_sdt8  = predict_sdt8
            vo.predict_edt8  = predict_edt8
            vo.model_name = 'CustomRegression'
            vo.store_nbr = store_nbr+1
            vo.family2 = family2 + 1

            train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)

            n_jobs = -1;
            seed = 1
            if vo.family2 == 32:  # SCHOOL AND OFFICE SUPPLIES
                etr = ExtraTreesRegressor(n_estimators=500, n_jobs=n_jobs, random_state=seed)
                rfr = RandomForestRegressor(n_estimators=500, n_jobs=n_jobs, random_state=seed)
                br1 = BaggingRegressor(base_estimator=etr, n_estimators=10, n_jobs=n_jobs, random_state=seed)
                br2 = BaggingRegressor(base_estimator=rfr, n_estimators=10, n_jobs=n_jobs, random_state=seed)
                model = VotingRegressor([('ExtraTrees', br1), ('RandomForest', br2)])
            else:
                ridge = make_pipeline(RobustScaler(), Ridge(alpha=31.0, random_state=seed))
                svr = make_pipeline(RobustScaler(), SVR(C=1.68, epsilon=0.09, gamma=0.07))
                model = VotingRegressor([('ridge', ridge), ('svr', svr)])

            model.fit(train_X, train_y.values.ravel())
            predict_y = pd.DataFrame(model.predict(test_X), index=test_X.index, columns=test_y.columns).clip(0.0)
            vo.mse = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
            vo.score = float("{:.2f}".format(1 - mean_squared_log_error(test_y, predict_y)))
            vo.test_y = test_y['sales'].tolist()
            vo.predict_y = predict_y['sales'].tolist()

            vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())
            _vo_list.append(vo.__dict__)
            print(f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list

if __name__ == '__main__':
    print(">>>> main")
    _trans = load_data.train_master2()
    execute(_trans,'x001d001y001m003','date8, month2, day2, onpromotion, transactions','20130101','20170730','20170801','20170815')
    execute(_trans,'x001d002y001m003','date8, month2, day2, onpromotion, transactions','20170101','20170730','20170801','20170815')
    df_qry = predict_analysis.scenario_score_rate()
