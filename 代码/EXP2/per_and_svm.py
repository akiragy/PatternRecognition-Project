import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier as skSGD
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

data_iris = np.loadtxt("data_iris.csv", delimiter=",", usecols=[0,1,2,3])
X_1, X_2, X_3 = data_iris[:50, :], data_iris[50:100, :], data_iris[100:, :]

def build_dataset(X1, X2, seed0=None):
    """生成分类器要求的数据格式"""
    random.seed(seed0)  # 设置随机数种子
    train_p_idx = random.sample(range(50), 25)  # 随机采样，p为X1, n为X2
    train_n_idx = random.sample(range(50), 25)
    random.seed(None)  # 清除随机数种子

    X1_train, X1_test = X1[train_p_idx, :], X1[[i for i in range(50) if i not in train_p_idx], :]
    X2_train, X2_test = X2[train_n_idx, :], X2[[i for i in range(50) if i not in train_n_idx], :]
    y1_train, y1_test = np.ones([25]), np.ones([25])
    y2_train, y2_test = -np.ones([25]), -np.ones([25])

    return np.concatenate([X1_train, X2_train],axis=0), \
           np.concatenate([X1_test, X2_test],axis=0), \
           np.concatenate([y1_train, y2_train],axis=0),\
           np.concatenate([y1_test, y2_test],axis=0)

X_train, X_test, y_train, y_test = build_dataset(X_1[:,:2], X_3[:,:2], seed0=1)

#训练集
plt.scatter(X_train[:25,0], X_train[:25,1])
plt.scatter(X_train[25:,0], X_train[25:,1])

clf = skSGD(max_iter=100, loss="perceptron", penalty="none", alpha=0.01)
#clf = SVC(C=1.0, kernel="linear")
clf.fit(X_train, y_train)

x_line = np.linspace(3,8,100)
#y_line = -(clf.coef_[0] * x_line + clf.intercept_) / clf.coef_[1]
y_line = -(clf.coef_[0][0] * x_line + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_line,y_line)

print("系数: ", clf.coef_, "\n截距: ", clf.intercept_, "\n迭代次数: ", clf.n_iter_)