import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
# import file_util as file_util
import seaborn as sns # For Data Visualization
import numpy as np

from feature import load_data

def pyplot_predict_scenario_store_family(t1,xlabel,ylabel,x1,y1,y2):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(10,5))
    plt.title(t1, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    plt.grid()
    # plt.xlim(0, 5)
    # plt.ylim(0.5, 10)
    plt.plot(x1, y1, color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
    plt.plot(x1, y2, color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')
    plt.show()

def pyplot_predict_scenario_store(df):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fig, axes = plt.subplots(7, 5, figsize=(20, 20))

    for family2 in range(33) :
        df2 = df[(df['family2'] == family2+1)]
        t1 = "(" + str(df2['store_nbr'].values[0]) + ")" + "(" + str(family2+1) + ")" + "(" + str(df2['score'].values[0]) + ")"
        x1 = list(range(len(df2['test_y'].values[0])))
        y1 = df2['test_y'].values[0]
        y2 = df2['predict_y'].values[0]
        axes[family2 // 5, family2 % 5].grid()
        axes[family2 // 5, family2 % 5].set(title=t1, xlabel='date', ylabel='sales')
        axes[family2 // 5, family2 % 5].plot(x1, y1, color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
        axes[family2 // 5, family2 % 5].plot(x1, y2, color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')

    fig.tight_layout()
    plt.show()

def pyplot_predict_store_family(df):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fig, axes = plt.subplots(3, 5, figsize=(20, 20))

    inx = 0
    for index, df2 in df.iterrows():
        t1 = "(" + str(df2['scenario_id']) + ")" + "(" + str(df2['store_nbr']) + ")" + "(" + str(df2['family2']) + ")" + "(" + str(df2['score']) + ")"
        x1 = list(range(len(df2['test_y'])))
        y1 = df2['test_y']
        y2 = df2['predict_y']
        axes[inx // 5, inx % 5].grid()
        axes[inx // 5, inx % 5].set(title=t1, xlabel='date', ylabel='sales')
        axes[inx // 5, inx % 5].plot(x1, y1, color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
        axes[inx // 5, inx % 5].plot(x1, y2, color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')
        inx = inx + 1

    fig.tight_layout()
    plt.show()

def pyplot_train_store(df):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fig, axes = plt.subplots(7, 5, figsize=(20, 20))

    for family2 in range(33) :
        df2 = df[(df['family2'] == family2+1)]
        t1 = "(" + str(df2['store_nbr'].values[0]) + ")" + "(" + str(family2 + 1) + ")"
        #x1 = list(range(len(df2['date'])))
        x1 = df2['date']
        y1 = df2['sales']
        axes[family2 // 5, family2 % 5].grid()
        axes[family2 // 5, family2 % 5].set(title=t1, xlabel='date', ylabel='sales')
        axes[family2 // 5, family2 % 5].plot(x1, y1, color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')

    fig.tight_layout()
    plt.show()

def pyplot_train_store_family(df, title, x, y):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.title(title)
    plt.plot(df[x], df[y], color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
    plt.show()

def pyplot_train_store_family_multi(df):
    plt.rcParams['font.family'] = 'Malgun Gothic'

    mpl.rc('font', size=14)
    mpl.rc('axes', titlesize=15)
    fig, axes = plt.subplots(nrows=3, ncols=2)
    # plt.plot(df['date'], df['sales'], color='b', alpha=0.6, marker='o', markersize=5, linestyle='-', ax=axes[0, 0])
    axes[0, 0].plot(df['date'], df['sales'], color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')
    axes[0, 0].set(xlabel='date', ylabel='sales')
    #sns.barplot(x='date', y='sales', data=df, ax=axes[0, 0])

    sns.boxplot(x='year4', y='sales', data=df, ax=axes[0, 1])
    sns.boxplot(x='season', y='sales', data=df, ax=axes[1, 0])
    sns.boxplot(x='month2', y='sales', data=df, ax=axes[1, 1])
    sns.boxplot(x='day2', y='sales', data=df, ax=axes[2, 0])
    sns.boxplot(x='day_of_week', y='sales', data=df, ax=axes[2, 1])

    # sns.barplot(x='year4', y='sales', data=df, ax=axes[0, 1])
    # sns.barplot(x='season', y='sales', data=df, ax=axes[1, 0])
    # sns.barplot(x='month2', y='sales', data=df, ax=axes[1, 1])
    # sns.barplot(x='day2', y='sales', data=df, ax=axes[2, 0])
    # sns.barplot(x='day_of_week', y='sales', data=df, ax=axes[2, 1])

    fig.tight_layout()
    fig.set_size_inches(10,9)
    plt.show()

def plotly_df(df, title, x, y):
    fig = px.line(df, x=x, y=y, title=title)
    fig.show()

#%%
###########################################################
# test
###########################################################
#%%
# df = load_data.train_master("train_master_all")
# df = df[(df['store_nbr'] == 1) & (df['family2'] == 4)]

#%%

#%%
#df2 = df[(df['store_nbr'] == 1)]
# df3 = df2[df2["sales"] > 0]

#%%
#fig = px.line(df2, x='day_of_week', y='sales', title='')
#fig.show()
#sns.distplot(df2["day_of_week"])
#sns.distplot(df2["sales"])
#sns.distplot(np.log(df3["sales"]))
#sns.barplot(x='day_of_week', y='sales', data=df2)
#sns.barplot(x='year4', y='sales', data=df2)
# fig, axes = plt.subplots(nrows=3, ncols=2)
# sns.barplot(x='season', y='sales', data=df2, ax=axes[0, 0])
# plt.title("Histogram of Total Bill")
# plt.show()

#%%

#plt.plot(df2['day_of_week'], df2['day_of_week'], color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
#plt.show()
#
# #%%
#
# path = "d:/lge/pycharm-projects/kaggle_store_sales/output/"
# df = file_util.read_jsons_to_pandas(path)
# #%%
# pyplot_predict_store_family(df2)
#  #%%
#
# #%%
# loop1 = list(range(len(df2['scenario_id'])))
# loop2 = list(range(len(df2)))
#
# # pyplot_store_train(df2)
# #%%
#
# plt.rcParams['font.family'] = 'Malgun Gothic'
# fig, axes = plt.subplots(3, 5, figsize=(20, 20))
#
# inx = 0
# for index, df3 in df2.iterrows():
#     t1 = "(" + str(df2['scenario_id'].values[0]) + ")" + "(" + str(df2['store_nbr'].values[0]) + ")" + "(" + str(df2['family2'].values[0]) + ")" + "(" + str(df2['score'].values[0]) + ")"
#     x1 = list(range(len(df3['test_y'])))
#     y1 = df3['test_y']
#     y2 = df3['predict_y']
#     axes[inx // 5, inx % 5].grid()
#     axes[inx // 5, inx % 5].set(title=t1, xlabel='date', ylabel='sales')
#     axes[inx // 5, inx % 5].plot(x1, y1, color='b', alpha=0.6, marker='o', markersize=5, linestyle='-')
#     axes[inx // 5, inx % 5].plot(x1, y2, color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')
#     inx = inx + 1
#
# fig.tight_layout()
# plt.show()
#
# #%%
# # store_nbr = 1; family2 = 4
# # df2 = df[(df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
# # title = "(" + str(store_nbr) + ")" + "(" + str(family2) + ")"
# # train_x = df2['date8']
# # train_y = df2['sales']
# # # predict_y = df2['predict_y'].values[0]
# # pyplot_03(title, 'date', 'sales', train_x, train_y)
# #
# # #%%
# # fig = px.line(df2, x='date', y='sales', title='Daily total sales of the stores')
# # fig.show()
# # #%%
# #
# # plotly_01(df2,'date','sales','Daily total sales of the stores')
# # path = "d:/lge/pycharm-projects/kaggle_store_sales/output/"
# # df = file_util.read_jsons_to_pandas(path)
# # df2 = df[(df['scenario_id'] == 'x001d001y001m001') & (df['store_nbr'] == 1)]
# # pyplot_02(df2)
