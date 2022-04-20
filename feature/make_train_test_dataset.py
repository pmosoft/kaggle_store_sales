from feature import load_data
from pandasql import sqldf


def sql(qry): print(sqldf(qry))

def store_family_df(trans):
    q = '''
    select cast(store_nbr as int) store_nbr, cast(family2 as int) family2
    from   trans
    group by store_nbr, family2
    order by store_nbr, family2
    '''
    return sqldf(q)

def query(trans, vo):
    trans2 = trans[(trans['store_nbr'] == vo.store_nbr) & (trans['family2'] == vo.family2)]

    q = f"select {vo.feature_col} from trans2 where date8 between '{vo.feature_sdt8}' and '{vo.feature_edt8}' order by date";  # print(q)
    train_X = sqldf(q)

    q = f"select {vo.predict_col} from trans2 where date8 between '{vo.feature_sdt8}' and '{vo.feature_edt8}' order by date";  # print(q)
    train_y = sqldf(q)
    # train_y.set_index(keys=['date8'], inplace=True, drop=True)

    q = f"select {vo.feature_col} from trans2 where date8 between '{vo.predict_sdt8}' and '{vo.predict_edt8}' order by date";  # print(q)
    test_X = sqldf(q)

    q = f"select {vo.predict_col} from trans2 where date8 between '{vo.predict_sdt8}' and '{vo.predict_edt8}' order by date";  # print(q)
    test_y = sqldf(q)
    # test_y.set_index(keys=['date8'], inplace=True, drop=True)

    return train_X, train_y, test_X, test_y

def query2(trans, vo):
    trans2 = trans[(trans['store_nbr'] == vo.store_nbr) & (trans['family2'] == vo.family2)]

    q = f"select {vo.feature_col}, {vo.predict_col} from trans2 where date8 between '{vo.feature_sdt8}' and '{vo.feature_edt8}' order by date";  # print(q)
    train = sqldf(q)

    q = f"select {vo.feature_col}, {vo.predict_col} from trans2 where date8 between '{vo.predict_sdt8}' and '{vo.predict_edt8}' order by date";  # print(q)
    test = sqldf(q)
    # test_y.set_index(keys=['date8'], inplace=True, drop=True)

    return train, test

# %%
# _trans = load_data.train_master("train_master_exist_sales")
#
# # %%
# df = store_family_df(_trans)

# _train_X, _train_y, _test_X, _test_y = s002(_trans,1,2)
# %%

def s001(trans):
    q = '''
    select replace(substr(date,1,10),'-','') as date8
         , substr(date,6,2)                  as month2
         , substr(date,9,2)                  as day2
         --, onpromotion                       as onpromotion
         , transactions                      as transactions
    from   trans
    where  store_nbr = '1' and family = 'BEVERAGES'
    and    replace(substr(date,1,10),'-','') <= '20170730'
    order by date
    '''
    train_X = sqldf(q)

    q = '''
    select sales
    from   trans
    where  store_nbr = '1' and family = 'BEVERAGES'
    and    replace(substr(date,1,10),'-','') <= '20170730'
    order by date
    '''
    train_y = sqldf(q)

    q = '''
    select replace(substr(date,1,10),'-','') as date8
         , substr(date,6,2)                  as month2
         , substr(date,9,2)                  as day2
         --, onpromotion                       as onpromotion
         , transactions                      as transactions
    from   trans
    where  store_nbr = '1' and family = 'BEVERAGES'
    and    replace(substr(date,1,10),'-','') between '20170801' and '20170815'
    order by date
    '''
    test_X = sqldf(q)

    q = '''
    select replace(substr(date,1,10),'-','') as date8
         , sales
    from   trans
    where  store_nbr = '1' and family = 'BEVERAGES'
    and    replace(substr(date,1,10),'-','') between '20170801' and '20170815'
    order by date
    '''
    test_y = sqldf(q)
    test_y.set_index(keys=['date8'], inplace=True, drop=True)

    return train_X, train_y, test_X, test_y

# %%

# _trans = load_data.train_master3()
#
# #%%
#
# sql("select min(date), max(date) from trans limit 2")
#
# sql("select min(date), max(date) from trans where replace(substr(date,1,10),'-','') <= '20170730' limit 2")

# %%

# _train_X, _train_y, _test_X, _test_y = s001(_trans)
# %%
