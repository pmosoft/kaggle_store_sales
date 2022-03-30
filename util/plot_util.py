import matplotlib.pyplot as plt

def pyplot_01(t1,xlabel,ylabel,x1,y1,y2):
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

def pyplot_02(df):
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

# path = "d:/lge/pycharm-projects/kaggle_store_sales/output/"
# df = file_util.read_jsons_to_pandas(path)
# df2 = df[(df['scenario_id'] == 'x001d001y001m001') & (df['store_nbr'] == 1)]
# pyplot_02(df2)
