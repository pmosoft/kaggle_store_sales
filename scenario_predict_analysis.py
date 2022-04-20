import util.file_util as file_util
import util.db_util as pdb

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
path = "d:/lge/pycharm-projects/kaggle_store_sales/output/"

qry = """
select scenario_id
     , count(*) cnt
     , count(case when score > 0.81 then 1 else null end) as s90       
     , count(case when score between 0.61 and 0.80 then 1 else null end) as s70       
     , count(case when score between 0.41 and 0.60 then 1 else null end) as s50       
     , count(case when score between 0.21 and 0.40 then 1 else null end) as s30      
     , count(case when score < 0.20 then 1 else null end) as s10
     --, max(scenario_desc) as scenario_desc
     --, max(feature_col ) as feature_col
     --, max(feature_sdt8) as feature_sdt8
     --, max(feature_edt8) as feature_edt8
     --, max(predict_col ) as predict_col
     --, max(predict_sdt8) as predict_sdt8
     --, max(predict_edt8) as predict_edt8
     , round(sum(fit_tm_sec)/60)  as fit_tm_min
from df
group by scenario_id
order by substr(scenario_id,13,4), substr(scenario_id,1,4), substr(scenario_id,5,4)
"""

###########################################################
# analysis
###########################################################
def scenario_score_rate() :
    df = file_util.read_jsons_to_pandas(path)
    df.drop(columns=["test_y", "predict_y"], inplace=True)
    df_qry = pdb.sql_df(qry, df)
    return df_qry

# scenario_score_rate()
###########################################################
# test
###########################################################
#
# #%%
# df = file_util.read_jsons_to_pandas(path)
# df.drop(columns=["test_y", "predict_y"], inplace=True)
# #%%
#
# df_qry = pdb.sql_df(qry, df)
#
# qry = """
# select sum(fit_tm_sec)
# from df
# where 1=1
# --and store_nbr = 1
# --and   family2 =4
# and   scenario_id = 'x001d001y001m001'
# """




