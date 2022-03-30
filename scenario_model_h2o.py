import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_log_error

import pandas as pd
import json

from feature import load_data
from feature import make_train_test_dataset
from util.vo import experiment_vo
import util.date_util as dt
import util.string_util as string_util
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
pliot = True
# pliot = False
###############################################################################
#                                      구현부
###############################################################################

###########################################################
# pliot
###########################################################
_trans = load_data.train_master2()
h2o.init(port=54321)
# h2o.init(port=54322)
h2o.no_progress()

# %%

if pliot:
    start_dtm = dt.get_now()

    vo = experiment_vo()
    vo.scenario_id = 'x001d001y001m004'
    vo.scenario_desc = '기본 feature autoML h2o 모델'
    vo.feature_col = 'date8, month2, day2, onpromotion, transactions'
    vo.feature_sdt8 = '20170101'
    vo.feature_edt8 = '20170730'
    vo.predict_col = 'sales'
    vo.predict_sdt8 = '20170801'
    vo.predict_edt8 = '20170815'
    vo.model_name = 'h2o'
    vo.store_nbr = 1
    vo.family2 = 4
    train, test = make_train_test_dataset.query2(_trans, vo)

    h2o_train = h2o.H2OFrame(train)
    h2o_test = h2o.H2OFrame(test)
    # aml = H2OAutoML(max_runtime_secs=30, exclude_algos=['XGBoost', 'StackedEnsemble'])
    # aml = H2OAutoML(max_models=5, max_runtime_secs=200, max_runtime_secs_per_model=40, seed=1234, exclude_algos=['StackedEnsemble'])
    aml = H2OAutoML(max_models=5, max_runtime_secs=10, max_runtime_secs_per_model=2, seed=1234, exclude_algos=['StackedEnsemble'])
    #aml = H2OAutoML(exclude_algos=['StackedEnsemble'])

    aml.train(x=string_util.str_to_list(vo.feature_col), y=vo.predict_col, training_frame=h2o_train)

    predict_y = h2o.as_list(aml.leader.predict(h2o_test))

    vo.mse = float("{:.2f}".format(mean_squared_log_error(test['sales'], predict_y)))
    vo.score = float("{:.2f}".format(1 - vo.mse))
    vo.test_y = test['sales'].tolist()
    vo.predict_y = predict_y['predict'].tolist()

    best_model = list(aml.leaderboard['model_id'].as_data_frame().iloc[:, 0])[0]
    vo.auto_ml_model = best_model[0:best_model.index("AutoML")].replace("grid", "").replace("_","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","").replace("8","").replace("9","")
    vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())


# %%

best_model2 = list(aml.leaderboard['model_id'].as_data_frame())

# #preds = aml.predict(test)
# #mdl = h2o.get_model('GBM_grid_1_AutoML_1_20220326_225519_model_32')
# preds = aml.leader.predict(h2o_test)
# # %%
#
#
# pdaaa = h2o.as_list(preds)
# lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
# log = aml.event_log
# info = aml.training_info
# aml.explain_row(frame=test, row_index=15, figsize=(8, 6))
# explain_model = aml.explain(frame=test, figsize=(8, 6))
# aaaa = best_model.model_performance(test)
# explain_model = aml.explain(frame=test, figsize=(8, 6))
# preds = aml.leader.predict(test)
# leaderboard = aml.leaderboard
#     #performance = aml.leader.model_performance(h2o_test_X)  # (Optional) Evaluate performance on a test set
#
#     print(leaderboard.head(rows=leaderboard.nrows))

# %%

###########################################################
# 모델 fit, predict
###########################################################

def execute(trans, scenario_id, feature_col, feature_sdt8, feature_edt8, predict_sdt8, predict_edt8):
    _vo_list = []
    work_dtm16 = dt.get_time_str16();
    work_dtm15 = dt.get_time_str15()
    # for store_nbr in range(54):
    #     for family2 in range(33):
    for store_nbr in range(54):
        for family2 in range(33):
            start_dtm = dt.get_now()

            vo = experiment_vo()
            vo.work_dtm16 = work_dtm16
            vo.scenario_id = scenario_id
            vo.scenario_desc = '기본 feature autoML h2o 모델'
            vo.feature_col = feature_col
            vo.feature_sdt8 = feature_sdt8
            vo.feature_edt8 = feature_edt8
            vo.predict_col = 'sales'
            vo.predict_sdt8 = predict_sdt8
            vo.predict_edt8 = predict_edt8
            vo.model_name = 'h2o'
            vo.store_nbr = store_nbr + 1
            vo.family2 = family2 + 1

            train, test = make_train_test_dataset.query2(_trans, vo)

            h2o_train = h2o.H2OFrame(train)
            h2o_test = h2o.H2OFrame(test)
            aml = H2OAutoML(max_runtime_secs=30, exclude_algos=['XGBoost', 'StackedEnsemble'])

            try:
                aml.train(x=string_util.str_to_list(vo.feature_col), y=vo.predict_col, training_frame=h2o_train,
                          leaderboard_frame=h2o_train)

                predict_y = h2o.as_list(aml.leader.predict(h2o_test))

                vo.mse = float("{:.2f}".format(mean_squared_log_error(test['sales'], predict_y)))
                vo.score = float("{:.2f}".format(1 - vo.mse))
                vo.test_y = test['sales'].tolist()
                vo.predict_y = predict_y['predict'].tolist()

                best_model = list(aml.leaderboard['model_id'].as_data_frame().iloc[:, 0])[0]
                vo.auto_ml_model = best_model[0:best_model.index("AutoML")].replace("grid", "").replace("_",
                                                                                                        "").replace("1",
                                                                                                                    "")
            except:
                print("Exception")

            vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())
            _vo_list.append(vo.__dict__)
            print(
                f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
            # plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_y['date8'], test_y['sales'], predict_y['sales'])
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list


if __name__ == '__main__':
    print(">>>>> main")
    # execute(_trans,'x001d001y001m004','date8, month2, day2, onpromotion, transactions','20130101','20170730','20170801','20170815')
    # execute(_trans,'x001d002y001m004','date8, month2, day2, onpromotion, transactions','20170101','20170730','20170801','20170815')
    # df_qry = predict_analysis.scenario_score_rate()
