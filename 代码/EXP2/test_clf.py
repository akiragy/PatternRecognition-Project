import numpy as np
import random
from ErrorCorrectingAlg import Perceptron
from ErrorCorrectingAlg import RelaxationPro
from ErrorCorrectingAlg import SVM as MySVM
from SGDAlgs import SGDClassifier

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

X_train, X_test, y_train, y_test = build_dataset(X_1, X_3, seed0=1)

class TestAlgs(object):
    """测试各个算法"""
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def eval(self):
        self.algs = {}
        self.algs[3] = Perceptron(margin=0, batch_size=-1)
        self.algs[4] = Perceptron(margin=0, batch_size=1, eta0=1)
        self.algs[5] = Perceptron(margin=1, batch_size=1)
        self.algs[6] = Perceptron(margin=0, batch_size=-1, learning_rate="1/k")
        self.algs[8] = RelaxationPro(margin=1, batch_size=-1)
        self.algs[9] = RelaxationPro(margin=1, batch_size=1, eta0=0.05)

        for key in self.algs:
            self.algs[key].fit(X_train, y_train)  # 训练
            y_pred = self.algs[key].predict(X_test)  # 测试
            print(key, int(np.ones_like(y_test)[y_pred * y_test < 0].sum()))


testAlgs = TestAlgs(X_train, X_test, y_train, y_test)
#testAlgs.eval()



#clf = Perceptron()
clf = SGDClassifier(margin=1, penalty="none", alpha=5)
#clf = MySVM(batch_size=10, eta0=1, alpha=0.1)
#clf = RelaxationPro(margin=1, batch_size=10, eta0=1, max_iter=1000)
clf.fit(X_train, y_train)
print("权向量为：", clf.coef_, "\n截距为：", clf.intercept_, "\n迭代次数为：", clf.i_iter)
print("预测为：", clf.predict(X_test) * y_test)
print(np.linalg.norm(clf.coef_))


