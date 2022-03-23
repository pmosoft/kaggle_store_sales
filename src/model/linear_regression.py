from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

class linear_regression:
    def __init__(self, train_X, train_y, test_X, test_y):
        #print("linear_regression.__init__")
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.predict_y = None
        self.predict_y_df = None
        self.model = LinearRegression()

    def fit(self):
        self.model.fit(self.train_X, self.train_y)

    def predict(self):
        self.predict_y = self.model.predict(self.test_X)
        #self.predict_y_df = pd.Series(self.predict_y, index=self.test_y.index.values)

    def score(self):
        # print("스코어 : ", self.model.score(self.test_X, self.test_y))
        # print("가중치(계수, 기울기 파라미터 W) :", self.model.coef_)
        # print("편향(절편 파라미터 b) :", self.model.intercept_)
        # print("훈련세트 점수: {:.2f}".format(self.model.score(self.train_X, self.train_y)))
        # print("테스트세트 점수: {:.2f}".format(self.model.score(self.test_X, self.test_y)))
        score = self.model.score(self.test_X, self.test_y)
        #print("스코어: {:.2f}".format(score))
        return "{:.2f}".format(score)

    def visual_scatter(self):
        plt.scatter(self.test_y, self.predict_y, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted ")
        plt.title("MULTIPLE LINEAR REGRESSION")
        plt.show()

    def visual_plot(self):
        # plt.figure(figsize=(8,5))
        plt.title('s001', fontsize=20)
        plt.xlabel('date', fontsize=10)
        plt.ylabel('sales', fontsize=10)
        plt.xticks(rotation=70)
        plt.yticks(rotation=0)
        plt.grid()
        #plt.xlim(0, 5)
        #plt.ylim(0.5, 10)
        plt.plot(self.test_y, color='r', alpha=0.6, marker='o', markersize=5, linestyle='--')
        plt.plot(self.predict_y, color='b', alpha=0.6, marker='o', markersize=5, linestyle='--')
        plt.legend(['real', 'predict'], fontsize=10)
        plt.show()
