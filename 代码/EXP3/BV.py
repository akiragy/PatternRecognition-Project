# -- coding: UTF-8 --
"""废弃"""
import numpy as np

class TestBV(object):
    def __init__(self, m, n):
        """m: 数据集数量，n: 数据集大小"""
        self.m = m
        self.n = n
        self.create_datasets()

    def create_a_dataset(self):
        x = np.random.uniform(-1, 1, self.n)
        ep = np.random.normal(0, 0.01 * np.sqrt(0.1), self.n)  # 噪声
        y = x ** 2 + ep
        return np.c_[x, x**2, y]

    def create_datasets(self):
        self.datasets = {}
        for i in range(self.m):
            self.datasets[i] = self.create_a_dataset()

    def calc_mse(self, deg, dataset):
        x, Fx, y = dataset[:,0], dataset[:,1], dataset[:,2]
        w = 0  # 参数
        if deg >= 0:
            w = np.polyfit(x, y, deg)
            p = np.poly1d(w)
            gx = p(x)
        elif deg == -0.5:
            gx = 0.5 * np.ones_like(y)
        elif deg == -1:
            gx = np.ones_like(y)

        mse = np.linalg.norm(gx - Fx) ** 2 / self.n
        return w, mse

    def estimate_MSE(self):
        #self.mse1, self.mse2, self.mse3, self.mse4 = [], [], [], []
        ws = {1:[], 2:[], 3:[], 4:[]}
        mses = {1:[], 2:[], 3:[], 4:[]}
        for i in range(self.m):
            for key, value in {1:-0.5, 2:-1, 3:1, 4:3}.items():
                w, mse = self.calc_mse(value, self.datasets[i])
                ws[key].append(w)
                mses[key].append(mse)
        self.MSE = {key: np.mean(value) for key, value in mses.items()}
        self.W = {key: np.mean(value,axis=0) for key, value in ws.items()}
        self.W[1], self.W[2] = [0.5], [1]

        mses = {1:[], 2:[], 3:[], 4:[]}
        mm = {1:[], 2:[], 3:[], 4:[]}
        for i in range(self.m):
            x, Fx = self.datasets[i][:,0], self.datasets[i][:,2]
            for key in [1,2,3,4]:
                p = np.poly1d(self.W[key])
                dgx = p(x)
                pp = np.poly1d(ws[1][i])
                gx = pp(x)
                mses[key].append(np.linalg.norm(dgx - Fx) ** 2 / self.n)
                mm[key].append(np.linalg.norm(dgx - gx) ** 2 / self.n)
        self.B2 = {key: np.mean(value) for key, value in mses.items()}
        self.VAR = {key: np.mean(value) for key, value in mm.items()}










if __name__ == "__main__":
    te = TestBV(m=100, n=10)
    te.estimate_MSE()


