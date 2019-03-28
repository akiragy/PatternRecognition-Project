# -- coding: UTF-8 --
import numpy as np
import matplotlib.pyplot as plt
import time

class TestBV(object):
    def __init__(self, m, n):
        """m: 数据集数量，n: 数据集大小"""
        self.m = m
        self.n = n
        self.create_datasets()
        self.mse_s = {}

    def create_datasets(self):
        #np.random.seed(9887)  # seed = 9887(10)
        self.datasets = {}
        for i in range(self.m):
            x = np.random.uniform(-1, 1, self.n)
            ep = np.random.normal(0, np.sqrt(0.1), self.n)  # 噪声
            y = x ** 2 + ep
            self.datasets[i] = np.c_[x, x**2, y]

    def fit(self, deg, dataset):
        """当deg>=0时，拟合deg阶多项式；当<0时，返回常数-deg"""
        x, Fx, y = dataset[:, 0], dataset[:, 1], dataset[:, 2]
        return np.polyfit(x, y, deg) if deg >= 0 else -deg

    def esti_BV(self, deg):
        ws = []  # 每一个数据集上的参数
        for i in range(self.m):
            w = self.fit(deg, self.datasets[i])
            ws.append(w)
        w_mean = np.mean(ws, axis=0)  # 参数在所有数据集上的期望

        bias2_s, var_s, mse_s = [], [], []
        for i in range(self.m):
            dataset = self.datasets[i]
            x, Fx, y = dataset[:, 0], dataset[:, 1], dataset[:, 2]  # Fx无噪声，y有噪声
            gx = np.poly1d(ws[i])(x)  # 当前数据集参数
            dgx = np.poly1d(w_mean)(x)  # 期望参数

            bias2_s.append(np.linalg.norm(dgx - y) ** 2 / self.n)  # 偏差的平方
            var_s.append(np.linalg.norm(gx - dgx) ** 2 / self.n)  # 方差
            mse_s.append(np.linalg.norm(gx - Fx) ** 2 / self.n)  # 好像不对

        self.mse_s[deg] = [bias2_s[i] + var_s[i] for i in range(self.m)]
        return np.mean(var_s), np.mean(bias2_s), np.mean(mse_s)

    def plot_hist(self):
        for i, key in enumerate(self.mse_s):
            plt.subplot(2, 2, i+1)
            plt.title("model"+str(i+1))
            plt.hist(self.mse_s[key], normed=True)
        plt.show()



if __name__ == "__main__":
    te = TestBV(m=100, n=10)

    print("{:<12}\t{:^6}\t{:^6}".format("模型", "平方偏差", "方差"))
    v, b, a = te.esti_BV(-0.5)
    print("{:<12}\t{:.6f}\t{:6f}".format("常数0.5", b, v))
    v, b, a = te.esti_BV(-1)
    print("{:<12}\t{:.6f}\t{:6f}".format("常数1", b, v))
    v, b, a = te.esti_BV(1)
    print("{:<12}\t{:.6f}\t{:6f}".format("一次多项式", b, v))
    v, b, a = te.esti_BV(3)
    print("{:<12}\t{:.6f}\t{:6f}".format("三次多项式", b, v))

    # v, b, a = te.esti_BV(3)
    # print(a-b-v)

    te.plot_hist()

