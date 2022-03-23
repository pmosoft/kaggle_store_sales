"""

create table experiment (
 work_dtm9      date          not null
,scenario_id    varchar(100)  not null
,scenario_desc  varchar(1000)
,feature_col    varchar(200)
,feature_cnt    int
,feature_desc   varchar(1000)
,model_name     varchar(100)
,model_cfg      varchar(1000)
,score          numeric
,mae            numeric
,mape           numeric
,mse            numeric
,fit_tm_sec     int
,primary key(work_dtm9, scenario_id)
)

select * from experiment

"""