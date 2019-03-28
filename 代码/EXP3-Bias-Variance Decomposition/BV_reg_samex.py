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
        np.random.seed(56)  # 2333
        self.datasets = {}
        x = np.random.uniform(-1, 1, self.n)
        for i in range(self.m):
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
        err_s = []
        for i in range(self.m):
            dataset = self.datasets[i]
            x, Fx, y = dataset[:, 0], dataset[:, 1], dataset[:, 2]  # Fx无噪声，y有噪声
            gx = np.poly1d(ws[i])(x)  # 当前数据集参数
            dgx = np.poly1d(w_mean)(x)  # 期望参数

            bias2_s.append(np.linalg.norm(dgx - Fx) ** 2 / self.n)  # 偏差的平方
            var_s.append(np.linalg.norm(gx - dgx) ** 2 / self.n)  # 方差
            mse_s.append(np.linalg.norm(gx - y) ** 2 / self.n)  # 好像不对

            err_s.append(np.linalg.norm(y - Fx) ** 2 / self.n)  # 添加的随机噪声

            #print(np.mean(err_s))
        self.mse_s[deg] = [bias2_s[i] + var_s[i]for i in range(self.m)]
        return np.mean(var_s), np.mean(bias2_s), np.mean(mse_s), np.mean(err_s)


    def plot_hist(self):
        fig_title = ["g(x)=0.5", "g(x)=1", "linear poly", "cubic poly"]
        for i, key in enumerate(self.mse_s):
            plt.subplot(2, 2, i+1)
            if key < 0:
                plt.xlim([-0.01,0.8])
                plt.ylim([0,100])
                plt.axvline(np.mean(self.mse_s[key]))
            else:
                plt.xlim([-0.01, 0.8])
                h = plt.hist(self.mse_s[key], histtype="step")
                # plt.plot(h[1][0:len(h[1])-1], h[0])

            plt.title(fig_title[i])
        plt.show()



if __name__ == "__main__":
    te = TestBV(m=100, n=500)

    print("{:<12}\t{:^6}\t{:^6}\t{:^12}\t{:^6}\t{:^6}".format("模型", "平方偏差", "方差", "噪声", "泛化误差" ,"mse-b-v-e"))
    v, b, a, e = te.esti_BV(-0.5)
    print("{:<12}\t{:.6f}\t{:6f}\t{:6f}\t{:6f}\t{:6f}".format("常数0.5", b, v, e, a ,a-b-v-e))
    v, b, a, e = te.esti_BV(-1)
    print("{:<12}\t{:.6f}\t{:6f}\t{:6f}\t{:6f}\t{:6f}".format("常数1", b, v, e, a, a-b-v-e))
    v, b, a, e = te.esti_BV(1)
    print("{:<12}\t{:.6f}\t{:6f}\t{:6f}\t{:6f}\t{:6f}".format("一次多项式", b, v, e, a, a-b-v-e))
    v, b, a, e = te.esti_BV(3)
    print("{:<12}\t{:.6f}\t{:6f}\t{:6f}\t{:6f}\t{:6f}".format("三次多项式", b, v, e, a, a-b-v-e))

    # v, b, a = te.esti_BV(3)
    # print(a-b-v)
    # print("{:<12}\t{:^6}\t{:^6}".format("模型", "平方偏差" ,"方差"))
    # v, b, a, e = te.esti_BV(-0.5)
    # print("{:<12}\t{:.6f}\t{:6f}".format("常数0.5", b, v))
    # v, b, a, e = te.esti_BV(-1)
    # print("{:<12}\t{:.6f}\t{:6f}".format("常数1", b, v))
    # v, b, a, e = te.esti_BV(1)
    # print("{:<12}\t{:.6f}\t{:6f}".format("一次多项式", b, v))
    # v, b, a, e = te.esti_BV(3)
    # print("{:<12}\t{:.6f}\t{:6f}".format("三次多项式", b, v))

    te.plot_hist()

