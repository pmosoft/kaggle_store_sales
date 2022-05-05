import util.file_util as file_util
import util.plot_util as plot_util
from feature import load_data


class Scenario_predict_visualization:
    def __init__(self):
        print('>>> __init__')
        self.predict_df = file_util.read_jsons_to_pandas('d:/lge/pycharm-projects/kaggle_store_sales/output/')
        self.trans = load_data.train_master("train_master_all")

    def plot_predict_scenario_store_family(self, scenario_id, store_nbr, family2):
        df = self.predict_df
        df2 = df[(df['scenario_id'] == scenario_id) & (df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        title = scenario_id + "(" + str(store_nbr) + ")" + "(" + str(family2) + ")" + "(" + str(df2['score'].values[0]) + ")"
        test_x = list(range(len(df2['test_y'].values[0])))
        test_y = df2['test_y'].values[0]
        predict_y = df2['predict_y'].values[0]
        plot_util.pyplot_predict_scenario_store_family(title, 'date', 'sales', test_x, test_y, predict_y)
        return df2

    def plot_predict_scenario_store(self, scenario_id, store_nbr):
        df = self.predict_df
        df2 = df[(df['scenario_id'] == scenario_id) & (df['store_nbr'] == store_nbr)]
        plot_util.pyplot_predict_scenario_store(df2)

    def plot_predict_store_family(self, store_nbr, family2):
        df = self.predict_df
        df2 = df[(df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        plot_util.pyplot_predict_store_family(df2)
        return df2

    def plot_train_store(self, store_nbr):
        df = self.trans
        df2 = df[(df['store_nbr'] == store_nbr)]
        plot_util.pyplot_train_store(df2)
        return df2

    def plot_train_store_family(self, store_nbr, family2):
        df = self.trans
        df2 = df[(df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        title = "(" + str(store_nbr) + ")" + "(" + str(family2) + ")"
        # plot_util.plotly_df(df2, title, 'date', 'sales')
        plot_util.pyplot_train_store_family(df2, title, 'date', 'sales')
        return df2

    def plot_train_store_family_multi(self, store_nbr, family2):
        df = self.trans
        df2 = df[(df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
        title = "(" + str(store_nbr) + ")" + "(" + str(family2) + ")"
        # plot_util.plotly_df(df2, title, 'date', 'sales')
        plot_util.pyplot_train_store_family_multi(df2)
        return df2

# %%
###########################################################
# execute
###########################################################
visual = Scenario_predict_visualization()

# %%
#visual.plot_train_store(1)
#df = visual.plot_train_store_family_multi(1, 1)
# visual.plot_predict_scenario_store_family('x001d001y001m004', 1, 4)
visual.plot_predict_scenario_store('x001f005d001y001t001m001c001', 10)
#visual.plot_predict_store_family(1,3)

# %%


# %%
###########################################################
# test
###########################################################
# %%
#
# df = visual.trans
# store_nbr = 1; family2 = 4
# df2 = df[(df['store_nbr'] == store_nbr) & (df['family2'] == family2)]
# title = "(" + str(store_nbr) + ")" + "(" + str(family2) + ")"
# train_x = df2['date8']
# train_y = df2['sales']
# # predict_y = df2['predict_y'].values[0]
# plot_util.pyplot_03(title, 'date', 'sales', train_x, train_y)
#

# %%
# aa = df[(df['store_nbr'] == 1)]
# df.columns
# df = file_util.read_jsons_to_pandas(path)
# df2 = df[(df['scenario_id'] == 'x001d001y001m001') & (df['store_nbr'] == 1) & (df['family2'] == 10)]
# print(df2['score'].head(1))
# title = df2['scenario_id'].values[0]+"("+str(df2['store_nbr'].values[0])+")"+"("+str(df2['family2'].values[0])+")"+"("+str(df2['score'].values[0])+")"
# test_x = list(range(len(df2['test_y'].values[0])))
# test_y = df2['test_y'].values[0]
# predict_y = df2['predict_y'].values[0]
# plot_util.pyplot_01(title,'date','sales', test_x, test_y, predict_y)
