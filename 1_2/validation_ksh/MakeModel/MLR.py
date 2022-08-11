from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np


check_col = ['isitu-LST', 'insitu-TG', 'GK2A-LST', 'insitu-TED0.05']     # isitu-LST
# check_col = ['isitu-LST','insitu-TG','GK2A-LST']     # isitu-LST
# check_col = ['isitu-LST', 'insitu-TG', 'GK2A-LST', 'insitu-TED0.05', 'insitu-TA']  # isitu-LST 2

df = pd.read_csv("remove_outlier.csv")
# df = pd.read_csv("remove_outlier.csv")[check_col]
# df = pd.read_csv("../과제2 결측제거/과제2 결측치 제거.csv")[check_col]     # best
# df = pd.read_csv("../과제2 결측제거 extract_col/과제2 결측치 제거.csv")[check_col]
# df = pd.read_csv("../과제2 결측제거 7&8제거/과제2 결측치 제거.csv")[check_col]
# df = pd.read_csv("../과제2 결측제거 5689/과제2 결측치 제거.csv")[check_col]
# df = pd.read_csv("../과제2 결측제거/merge/202107_merge.csv")[check_col]


def LR(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    model = LinearRegression()
    model.fit(x, y)

    w1 = model.coef_[0][0]
    w2 = model.coef_[0][1]
    w3 = model.coef_[0][2]
    b = model.intercept_[0]

    result = (w1 * x1 + w2 * x2 + w3 * x3 + b).reshape(-1, 1)
    # result = (w1 * x1 + w2 * x2 + b).reshape(-1, 1)

    print('w1: ', model.coef_[0][0], ", w2: ", model.coef_[0][1], ", w3: ", model.coef_[0][2], ", b:", model.intercept_[0])
    # print('w1: ', model.coef_[0][0], ", w2: ", model.coef_[0][1], ", b:", model.intercept_[0])

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1, x2, y)
    # ax.plot_surface(x1, x2, result, color="red")
    # plt.suptitle("LR function", size=24)
    # plt.title('w1=' + str(round(w1, 3)) + 'w2=' + str(round(w2, 3)) + ', b=' + str(round(b, 3)))
    # plt.show()

    return w1, w2, w3, b, result
    # return w1, w2, b, result


def MLR(x, y, epochs=5000, learning_rate=0.00000001):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    w1 = 0.0
    w2 = 0.0
    w3 = 0.0
    b = 0.0

    n = len(x)

    for i in range(epochs):
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = np.sum((hypothesis - y) ** 2) / n

        gradient_w1 = np.sum((w1 * x1 + w2 * x2 + w3 * x3 - y + b) * 2 * x1) / n
        gradient_w2 = np.sum((w1 * x1 + w2 * x2 + w3 * x3 - y + b) * 2 * x2) / n
        gradient_w3 = np.sum((w1 * x1 + w2 * x2 + w3 * x3 - y + b) * 2 * x3) / n
        gradient_b = np.sum((w1 * x1 + w2 * x2 + w3 * x3 - y + b) * 2) / n

        w1 -= learning_rate * gradient_w1
        w2 -= learning_rate * gradient_w2
        w3 -= learning_rate * gradient_w3
        b -= learning_rate * gradient_b

        if i % 100 == 0:
            print('Epoch ({:10d}/{:10d}) cost: {:10f}, W1: {:10f}, W2: {:10f}, W3: {:10f}, b:{:10f}'.
                  format(i, epochs, cost, w1, w2, w3, b))

    result = (w1 * x1 + w2 * x2 + w3 * x3 + b).reshape(-1, 1)

    return w1, w2, w3, b, result


y = df[check_col[0]].values.reshape(-1, 1)
x1 = df[check_col[1]].values.reshape(-1, 1)
x2 = df[check_col[2]].values.reshape(-1, 1)
x3 = df[check_col[3]].values.reshape(-1, 1)

data = np.concatenate((x1, x2, x3), axis=1)
# data = np.concatenate((x1, x2), axis=1)

print("LR")
_, _, _, _, res = LR(data, y)
# _, _, _, res = LR(data, y)
print("결정계수: ", r2_score(y, res))
print("상관계수: \n", df.corr())
print("MSE: ", mean_squared_error(y, res))
print("RMSE: ", np.sqrt(mean_squared_error(y, res)))
print("MAE: ", mean_absolute_error(y, res))


# print("\nnMLR")
# _, _, _, _, res = MLR(data[:10000], y[:10000], epochs=100, learning_rate=0.001)
# print("결정계수: ", r2_score(y, res))
# print("상관계수: \n", df.corr())
# print("MSE: ", mean_squared_error(y, res))
# print("RMSE: ", np.sqrt(mean_squared_error(y, res)))
# print("MAE: ", mean_absolute_error(y, res))