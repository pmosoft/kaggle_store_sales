###############################################
# loading
###############################################
import pandas as pd

def train_master(tab_nm):
    print(f'load_data.train_master({tab_nm})')
    path = "d:/lge/pycharm-projects/kaggle_store_sales/input/"
    trans = pd.read_csv(f'{path}/{tab_nm}.csv',
                        usecols=['store_nbr', 'family', 'family2', 'date', 'date8', 'year4', 'season', 'day_of_week', 'month2', 'day2', 'onpromotion', 'transactions', 'sales'],
                        dtype={
                            # date
                            'date8': 'category',
                            'year4': 'uint32',
                            'season': 'uint32',
                            'day_of_week': 'uint32',
                            'month2': 'category',
                            'day2': 'category',
                            'store_nbr': 'uint32',
                            'family': 'category',
                            'family2': 'uint32',
                            'onpromotion': 'uint32',
                            'transactions': 'uint32',
                            'sales': 'uint32',
                        },
                        parse_dates=['date'], infer_datetime_format=True
                        )
    return trans

#_trans = train_master2("train_master_all")

