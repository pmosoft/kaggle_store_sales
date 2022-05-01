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
        self.mae = 0.0           # mae
        self.mse = 0.0           # mse
        self.rmse = 0.0          # rmse
        self.msle = 0.0          # msle
        self.rmsle = 0.0         # rmsle
        self.r2 = 0.0            # r2
        self.score = 0.0         # 스코어
        self.fit_tm_sec = 0      # fit수행시간
        self.file_nm = ''        # 파일명           : 'x001d001y001m001_2022-03-13_1400'
        self.memo = ''           # 메모

def vo_print(vo):
    print(vo.work_dtm16)

'''

schema = StructType([
    StructField("work_dtm16"    , StringType()),
    StructField("scenario_id"   , StringType()),
    StructField("scenario_desc" , StringType()),
    StructField("feature_src"   , StringType()),
    StructField("feature_col"   , StringType()),
    StructField("feature_sdt8"  , StringType()),
    StructField("feature_edt8"  , StringType()),
    StructField("feature_desc"  , StringType()),
    StructField("predict_col"   , StringType()),
    StructField("predict_sdt8"  , StringType()),
    StructField("predict_edt8"  , StringType()),
    StructField("auto_ml_model" , StringType()),
    StructField("model_name"    , StringType()),
    StructField("model_cfg"     , StringType()),
    StructField("store_nbr"     , StringType()),
    StructField("family2"       , StringType()),
    StructField("test_y"        , StringType()),
    StructField("predict_y"     , StringType()),
    StructField("mae"           , StringType()),
    StructField("mse"          , StringType()),
    StructField("rmse"           , StringType()),
    StructField("msle"           , StringType()),
    StructField("rmsle"           , StringType()),
    StructField("r2"           , StringType()),
    StructField("score"         , StringType()),
    StructField("fit_tm_sec"    , StringType()),
    StructField("file_nm"       , StringType()),
    StructField("memo"          , StringType())
    ])
    
'''

