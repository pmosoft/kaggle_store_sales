class experiment_vo:
    def __init__(self):
        self.work_dtm16 = ''     # 작업일시         : '2022-03-13 14:00'
        self.scenario_id = ''    # 시나리오ID       : 'x001c001d001y001m001c001'
        self.scenario_desc = ''  # 시나리오설명     : '기본 feature 기본 모델 회귀 only'
        self.feature_src  = ''   # feature 원천     : train_master_all, traion_master_store_dates, traion_master_exist_sales
        self.feature_col = ''    # feature컬럼      : 'date8, month2, day2, onpromotion, transactions'
        self.feature_sdt8 = ''   # feature시작일자  : '20170101'
        self.feature_edt8 = ''   # feature종료일자  : '20170730'
        self.feature_desc = ''   # feature설명
        self.predict_col = ''    # predict 컬럼     : 'date8, sales'
        self.predict_sdt8 = ''   # predict시작일자  : '20170801'
        self.predict_edt8 = ''   # predict종료일자  : '20170815'
        self.auto_ml_model = ''  # auto_ml best model
        self.model_name = ''     # 모델명           : 'LinearRegression'
        self.model_cfg = ''      # 모델cfg
        self.store_nbr = 0       # 상점             : 1
        self.family2 = 0         # 품목             : 1
        self.test_y = []         # 실제값 list
        self.predict_y = []      # 예측치 list
        self.score = 0.0         # 스코어
        self.mae = 0.0           # mae
        self.mape = 0.0          # mape
        self.mse = 0.0           # mse
        self.fit_tm_sec = 0      # fit수행시간
        self.file_nm = ''        # 파일명           : 'x001d001y001m001_2022-03-13_1400'
        self.meno = ''           # 메모

def vo_print(vo):
    print(vo.work_dtm16)

