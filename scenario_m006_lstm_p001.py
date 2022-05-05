from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

import tensorflow as tf

import json
import numpy as np

from feature import load_data
from feature import make_train_test_dataset
from model.experiment_vo import experiment_vo
import model.scenario_id_code as scd

import util.date_util as dt
import util.model_util as model_util
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
#pliot = False
###############################################################################
#                                      구현부
###############################################################################

###########################################################
# pliot
###########################################################
# %%
if pliot:
    start_dtm = dt.get_now()
    scenario_id = 'x001f006d001y001t001m006c001'
    scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_col, predict_sdt8, predict_edt8, model_name = scd.get_code_name(scenario_id)
    _trans = load_data.train_master(tab_nm)
    # %%
    # tf.debugging.set_log_device_placement(True)
    # with tf.device("CPU"):
    with tf.device("GPU"):
        start_dtm = dt.get_now()

        vo = experiment_vo()
        vo.scenario_id = scenario_id
        vo.feature_col = feature_col
        vo.feature_sdt8 = feature_sdt8
        vo.feature_edt8 = feature_edt8
        vo.predict_col = predict_col
        vo.predict_sdt8 = predict_sdt8
        vo.predict_edt8 = predict_edt8
        vo.model_name = model_name
        vo.meno = 'date8 삭제, 장기간 더 좋은 결과, 돌리때마다 다른 결과, early_stop_patience, batch_size도 결과에 영향 존재'
        vo.store_nbr = 1
        vo.family2 = 1
        model_cfg = {
            'window_size': 20,
            'hidden_layer_cnt': 128,
            'hidden_layer_activation': 'tanh',
            'output_layer_activation': 'linear',
            'loss': 'mse',
            'optimizer': 'adam',
            'early_stop_patience': 30,
            'epochs': 100,
            'batch_size': 8
        }
        vo.model_cfg = model_cfg

        train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)
        scaler = MinMaxScaler()
        train_X1 = scaler.fit_transform(train_X)
        train_y1 = scaler.fit_transform(train_y)

        window_size = model_cfg['window_size']
        X, Y = model_util.make_sequence_dataset(train_X1, train_y1, window_size)

        split = -1 * (dt.get_diff_time_day(dt.get_time_from_str8(vo.predict_sdt8), dt.get_time_from_str8(vo.predict_edt8)) + 1)
        train_X2 = X[0:split]
        train_y2 = Y[0:split]

        test_X2 = X[split:]
        test_y2 = Y[split:]

        model = Sequential()
        model.add(LSTM(model_cfg['hidden_layer_cnt'], activation=model_cfg['hidden_layer_activation'], input_shape=train_X2[0].shape))
        model.add(Dense(1, activation=model_cfg['output_layer_activation']))
        aa = model.summary()

        model.compile(loss=model_cfg['loss'], optimizer=model_cfg['optimizer'], metrics=['mse'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=model_cfg['early_stop_patience'])
        model.fit(train_X2, train_y2, validation_data=(test_X2, test_y2), epochs=model_cfg['epochs'], batch_size=model_cfg['batch_size'], callbacks=[early_stopping])

        predict_y = model.predict(test_X2)
        predict_y = scaler.inverse_transform(predict_y)

        vo.mae = float("{:.2f}".format(mean_absolute_error(test_y, predict_y)))
        vo.mse = float("{:.2f}".format(mean_squared_error(test_y, predict_y)))
        vo.rmse = float("{:.2f}".format(np.sqrt(mean_squared_error(test_y, predict_y))))
        vo.msle = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
        vo.rmsle = float("{:.2f}".format(np.sqrt(mean_squared_log_error(test_y, predict_y))))
        vo.r2 = float("{:.2f}".format(r2_score(test_y, predict_y)))
        vo.score = float("{:.2f}".format(1 - vo.msle))

        vo.test_y = test_y['sales'].tolist()
        vo.predict_y = predict_y.tolist()
        vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())
        score = vo.score
        rmsle = vo.rmsle
        fit_tm_sec = vo.fit_tm_sec

        # Save - Load
        # from pathlib import Path
        # import os
        # model_structure = model.to_json()
        # path = f"d:/lge/pycharm-projects/kaggle_store_sales/output/model/{vo.scenario_id}/"
        # os.makedirs(path)
        # f = Path(f"{path}/model_structure.json")
        # f.write_text(model_structure)
        # model.save_weights(f"{path}/save_weights.h5")
        # model.save(f"{path}/save.h5")
        # del model
        # from keras.models import load_model
        # model = load_model(f"{path}/save.h5")
        # predict_y = model.predict(test_X2)
        # predict_y = scaler.inverse_transform(predict_y)


# %%
###########################################################
# 모델 fit, predict
###########################################################

def execute(scenario_id, scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_sdt8, predict_edt8, model_name, model_cfg):
    _trans = load_data.train_master(tab_nm)

    _vo_list = []
    work_dtm16 = dt.get_time_str16();
    work_dtm15 = dt.get_time_str15()

    store_family_df = make_train_test_dataset.store_family_df(_trans)
    for i in store_family_df.index:
        store_nbr = int(store_family_df['store_nbr'][i])
        family2 = int(store_family_df['family2'][i])

        start_dtm = dt.get_now()

        # tf.debugging.set_log_device_placement(True)
        # with tf.device("CPU"):
        with tf.device("GPU"):

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
            vo.model_cfg = model_cfg
            vo.memo = 'date8 삭제, 장기간 더 좋은 결과, 돌리때마다 다른 결과, early_stop_patience, batch_size도 결과에 영향 존재'

            train_X, train_y, test_X, test_y = make_train_test_dataset.query(_trans, vo)
            scaler = MinMaxScaler()
            train_X1 = scaler.fit_transform(train_X)
            train_y1 = scaler.fit_transform(train_y)

            window_size = model_cfg['window_size']
            X, Y = model_util.make_sequence_dataset(train_X1, train_y1, window_size)

            split = -1 * (dt.get_diff_time_day(dt.get_time_from_str8(vo.predict_sdt8), dt.get_time_from_str8(vo.predict_edt8)) + 1)
            train_X2 = X[0:split]
            train_y2 = Y[0:split]

            test_X2 = X[split:]
            test_y2 = Y[split:]

            model = Sequential()
            model.add(LSTM(model_cfg['hidden_layer_cnt'], activation=model_cfg['hidden_layer_activation'], input_shape=train_X2[0].shape))
            model.add(Dense(1, activation=model_cfg['output_layer_activation']))
            aa = model.summary()

            model.compile(loss=model_cfg['loss'], optimizer=model_cfg['optimizer'], metrics=['mse'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=model_cfg['early_stop_patience'])
            model.fit(train_X2, train_y2, validation_data=(test_X2, test_y2), epochs=model_cfg['epochs'], batch_size=model_cfg['batch_size'], callbacks=[early_stopping])

            predict_y = model.predict(test_X2)
            predict_y = scaler.inverse_transform(predict_y)

            try:
                vo.mse = float("{:.2f}".format(mean_squared_log_error(test_y, predict_y)))
                vo.score = float("{:.2f}".format(1 - vo.mse))
            except:
                pass

            vo.test_y = test_y['sales'].tolist()
            vo.predict_y = predict_y.tolist()
            vo.fit_tm_sec = dt.get_diff_time_microseconds(start_dtm, dt.get_now())

        _vo_list.append(vo.__dict__)
        print(f"scenario_id={vo.scenario_id} store_nbr={vo.store_nbr} family2={vo.family2} score={vo.score} fit_tm_sec={vo.fit_tm_sec} ")
        # plot_util.pyplot_01(vo.scenario_desc, 'date', 'sales', test_y['date8'], test_y['sales'], predict_y['sales'])
    json.dump(_vo_list, open(f"{path}/{_vo_list[0]['scenario_id']}.json", 'w'))
    vo_list = _vo_list


if __name__ == '__main__':
    print(">>>> main")

    # model_cfg = {
    #     'window_size': 20,
    #     'hidden_layer_cnt': 128,
    #     'hidden_layer_activation': 'tanh',
    #     'output_layer_activation': 'linear',
    #     'loss': 'mse',
    #     'optimizer': 'adam',
    #     'early_stop_patience': 15,
    #     'epochs': 100,
    #     'batch_size': 8
    # }
    # execute('x002f003d003y001m006c001', 'sales 존재-기본 feature3-전기간-lstm 모델', 'train_master_exist_sales', 'month2, day2, day_of_week, onpromotion, transactions', '20130101', '20170815', '20170801', '20170815', 'LSTM', model_cfg)
    #
    # df_qry = predict_analysis.scenario_score_rate()
