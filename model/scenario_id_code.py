
scenario_id_code = {
    # get_code_name('x001f004d001y001t001m001c001p001')
    'x001': 'train_master_all',
    'x002': 'train_master_exist_sales',
    'x003': 'train_master_store_dates',
    'f001': 'date8, month2, day2, onpromotion, transactions',
    'f002': 'date8, month2, day2, day_of_week, onpromotion, transactions',
    'f003': 'month2, day2, day_of_week, onpromotion, transactions',
    'f004': 'month2, day2, day_of_week, onpromotion',
    'f005': 'date8, month2, day2, day_of_week, onpromotion',
    'f006': 'year4, season, day_of_week, onpromotion',
    'f007': 'year4, month2, day2, season, day_of_week, onpromotion',
    'f008': 'date8, year4, month2, day2, season, day_of_week, onpromotion',
    'f009': 'date8, year4, month2, day2, day_of_week, onpromotion',
    'f010': 'date8, season, month2, day2, day_of_week, onpromotion',
    'd001': '20130101,20170730',
    'd002': '20170101,20170730',
    'd003': '20130101,20170815',
    'd004': '20170101,20170815',
    'y001': 'sales',
    't001': '20170801,20170815',
    't002': '20170815,20170831',
    'm001': 'LinearRegression',
    'm002': 'Ridge',
    'm003': 'CustomRegressor',
    'm004': 'H2O',
    'm005': 'Pycaret',
    'm006': 'LSTM',
    'p001': 'shphi',  # 피승현 : 41-54
    'p002': 'hshong', # 홍혜선 : 01-10
    'p003': 'kikim',  # 김건일 : 21-30
    'p004': 'jhlee',  # 이준호 : 11-20
    'p005': 'jujung', # 정준하 : 31-40
    'etc': ''
}
def get_code_name(cd):
    s = scenario_id_code
    # print(cd[0:4])
    # print(cd[4:8])
    # print(cd[8:12])
    # print(s[cd[8:12]].split(',')[0])
    # print(cd[12:16])
    # print(cd[16:20])

    tab_nm = s[cd[0:4]]
    feature_col = s[cd[4:8]]
    feature_sdt8 = s[cd[8:12]].split(',')[0]
    feature_edt8 = s[cd[8:12]].split(',')[1]
    predict_col = s[cd[12:16]]
    predict_sdt8 = s[cd[16:20]].split(',')[0]
    predict_edt8 = s[cd[16:20]].split(',')[1]
    model_name = s[cd[20:24]]

    #scenario_desc = f'tab_nm[{tab_nm}] feature_col[{feature_col}] feature_date[{s[cd[8:12]]}] predict_col[{predict_col}] predict_date[{s[cd[16:20]]}] model_name[{model_name}] '
    scenario_desc = f'[{tab_nm}][{feature_col}][{s[cd[8:12]]}][{predict_col}][{s[cd[16:20]]}][{model_name}]'
    print(scenario_desc)

    return scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_col, predict_sdt8, predict_edt8, model_name

#scenario_desc, tab_nm, feature_col, feature_sdt8, feature_edt8, predict_col, predict_sdt8, predict_edt8, model_name = get_code_name('x001f004d001y001t001m001c001')

