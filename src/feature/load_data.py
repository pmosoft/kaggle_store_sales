###############################################
# loading
###############################################
import pandas as pd

def train_master2():
    print('load_data.train_master2')
    path = "d:/lge/pycharm-projects/kaggle_store_sales/input/"
    trans = pd.read_csv(f'{path}/train_master2.csv',
                        usecols=['store_nbr', 'family', 'family2', 'date', 'date8', 'month2', 'day2', 'day_of_week','onpromotion', 'transactions', 'sales'],
                        dtype={
                            # date
                            'date8': 'category',
                            'month2': 'category',
                            'day2': 'category',
                            'day_of_week': 'category',
                            'store_nbr': 'uint32',
                            'family': 'category',
                            'family2': 'uint32',
                            'onpromotion': 'uint32',
                            'transactions': 'uint32',
                            'sales': 'float32',
                        },
                        parse_dates=['date'], infer_datetime_format=True
                        )
    return trans

#_trans = train_master2()

def train_master3():
    print('>>> load_data.train_master3')
    path = "d:/lge/pycharm-projects/kaggle_store_sales/input/"
    trans = pd.read_csv(f'{path}/train_master3.csv',
                        usecols=['store_nbr', 'family', 'family2', 'date','onpromotion', 'transactions', 'sales'],
                        dtype={
                            # date
                            'store_nbr': 'uint32',
                            'family': 'category',
                            'family2': 'uint32',
                            'onpromotion': 'uint32',
                            'transactions': 'uint32',
                            'sales': 'float32',
                        },
                        parse_dates=['date'], infer_datetime_format=True
                        )
    return trans

# _trans = train_master3()
