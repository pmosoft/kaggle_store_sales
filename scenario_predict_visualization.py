import util.file_util as file_util
import util.plot_util as plot_util


class Scenario_predict_visualization:
    def __init__(self):
        print('>>> __init__')
        self.predict_df = file_util.read_jsons_to_pandas('d:/lge/pycharm-projects/kaggle_store_sales/output/')

    def plot_store_family(self, scenario_id, store_nbr, family2):
        df = self.predict_df
        df2 = df[(df['scenario_id'] == scenario_id) & (df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        title = scenario_id + "(" + str(store_nbr) + ")" + "(" + str(family2) + ")" + "(" + str(df2['score'].values[0]) + ")"
        test_x = list(range(len(df2['test_y'].values[0])))
        test_y = df2['test_y'].values[0]
        predict_y = df2['predict_y'].values[0]
        plot_util.pyplot_01(title, 'date', 'sales', test_x, test_y, predict_y)

    def plot_store_family_train(self, scenario_id, store_nbr, family2):
        df = self.predict_df
        df2 = df[(df['scenario_id'] == scenario_id) & (df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        title = scenario_id + "(" + str(store_nbr) + ")" + "(" + str(family2) + ")" + "(" + str(df2['score'].values[0]) + ")"
        test_x = list(range(len(df2['test_y'].values[0])))
        test_y = df2['test_y'].values[0]
        predict_y = df2['predict_y'].values[0]
        plot_util.pyplot_01(title, 'date', 'sales', test_x, test_y, predict_y)

    def plot_store(self, scenario_id, store_nbr):
        df = self.predict_df
        df2 = df[(df['scenario_id'] == scenario_id) & (df['store_nbr'] == store_nbr)]
        plot_util.pyplot_02(df2)

# %%
###########################################################
# execute
###########################################################
# visual = Scenario_predict_visualization()

# %%
# visual.plot_store_family('x001d001y001m004', 1, 7)
# visual.plot_store('x001d001y001m003', 2)
# %%
###########################################################
# test
###########################################################
# %%
# df = file_util.read_jsons_to_pandas(path)
# df2 = df[(df['scenario_id'] == 'x001d001y001m001') & (df['store_nbr'] == 1) & (df['family2'] == 10)]
# print(df2['score'].head(1))
# title = df2['scenario_id'].values[0]+"("+str(df2['store_nbr'].values[0])+")"+"("+str(df2['family2'].values[0])+")"+"("+str(df2['score'].values[0])+")"
# test_x = list(range(len(df2['test_y'].values[0])))
# test_y = df2['test_y'].values[0]
# predict_y = df2['predict_y'].values[0]
# plot_util.pyplot_01(title,'date','sales', test_x, test_y, predict_y)
