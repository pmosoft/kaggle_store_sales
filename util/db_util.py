from pandasql import sqldf

def sql(qry): print(sqldf(qry))

def sql(qry):
    print(sqldf(qry))

def sql_df(qry, df):
    df_qry = sqldf(qry, {'df': df})
    print(df_qry)
    return df_qry

def sql_dfs(qry, df):
    print(sqldf(qry, {'df':df}))



