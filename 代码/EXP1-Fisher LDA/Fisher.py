# -- coding: UTF-8 --
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats

data = np.loadtxt("data.csv", delimiter=',')
X_1, X_2, X_3 = data[:10, :], data[10:20, :], data[20:, :]  # 三个类


# (a)
def fisher(X1, X2):
    """对属于c1的数据X1和属于c2的数据进行fisher判别分析"""
    m1, m2 = np.mean(X1, axis=0, keepdims=True), np.mean(X2, axis=0, keepdims=True)  # 两个类分别的均值
    S1, S2 = np.dot(np.transpose(X1 - m1), X1 - m1), np.dot(np.transpose(X2 - m2), X2 - m2)  # 分别的散度矩阵
    Sw = S1 + S2  # 类内散布矩阵
    w = np.dot(np.linalg.inv(Sw), np.transpose(m1 - m2))  # 投影方向（未标准化）
    return w


# (b)
w = fisher(X_2, X_3)
print("对于w2和w3的最优方向为: ", w)


# (c)
def plot_w(X1, X2, w):
    """没啥可说的"""
    ax = plt.subplot(111, projection='3d')
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2])  # 画投影前的点
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2])

    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 2.])
    ax.set_zlim([-0.5, 3.5])

    xline = np.array([-0.5, 1.5])
    yline = xline * w[1][0] / w[0][0]
    zline = xline * w[2][0] / w[0][0]
    ax.plot(xline, yline, zline)  # 画投影方向


    w_norm = w / np.linalg.norm(w,2)  # 标准化后的投影方向（必要）
    Y1, Y2 = np.dot(np.dot(X1, w_norm), np.transpose(w_norm)), np.dot(np.dot(X2, w_norm), np.transpose(w_norm))
    ax.scatter(Y1[:, 0], Y1[:, 1], Y1[:, 2])  # 画投影后的点
    ax.scatter(Y2[:, 0], Y2[:, 1], Y2[:, 2])

    for i in range(10):
        ax.plot([X1[i][0], Y1[i][0]], [X1[i][1], Y1[i][1]], [X1[i][2], Y1[i][2]], "b--")  # 画样本点与投影点的连线
        ax.plot([X2[i][0], Y2[i][0]], [X2[i][1], Y2[i][1]], [X2[i][2], Y2[i][2]], "k--")

    plt.show()

plot_w(X_2, X_3 ,w)


# (d) and (e)
def density_est(X1, X2, w):
    """用高斯分布拟合两类投影后的样本点"""
    w_norm = w / np.linalg.norm(w, 2)  # 标准化后的投影方向（非必要）
    Y1, Y2 = np.dot(X1, w_norm), np.dot(X2, w_norm)  # 投影后的点
    mu1, mu2 = np.mean(Y1, keepdims=True), np.mean(Y2, keepdims=True)  # 两类的均值
    sigma1, sigma2 = np.dot(np.transpose(Y1 - mu1), Y1 - mu1) / Y1.shape[0], \
                     np.dot(np.transpose(Y2 - mu2), Y2 - mu2) / Y2.shape[0]  # 两类的方差

    plt.scatter(Y1, np.zeros([10, 1]), label="w2")  # 画样本点
    plt.scatter(Y2, np.zeros([10, 1]), label="w3")

    t = np.linspace(min(min(Y1), min(Y2)) - 0.5, max(max(Y1), max(Y2)) + 0.5, 100)
    p1, p2 = np.transpose(scipy.stats.norm(mu1, np.sqrt(sigma1)).pdf(t)),\
             np.transpose(scipy.stats.norm(mu2, np.sqrt(sigma2)).pdf(t))
    plt.plot(t, p1, label="p(x|w2)")  #画pdf曲线
    plt.plot(t, p2, label="p(x|w3)")

    plt.ylim(-0.1, max(max(p1), max(p2)) + 0.5)

    #数值算法求解分界点
    t_ = np.linspace(min(min(Y1), min(Y2)) - 0.5, max(max(Y1), max(Y2)) + 0.5, 10000)  # x的搜索空间
    p1_, p2_ = np.transpose(scipy.stats.norm(mu1, np.sqrt(sigma1)).pdf(t_)),\
             np.transpose(scipy.stats.norm(mu2, np.sqrt(sigma2)).pdf(t_))
    p_delta = p1_ - p2_  # 两个pdf的差

    point = []  # 记录交点
    for i in range(len(p_delta) - 1):
        if p_delta[i+1] * p_delta[i] < 0:
            point.append(i+1)

    assert len(point) <= 2  # 至多有两个交点

    if len(point) == 1:
        # 有一个交点时
        bound = min(t_) + point[0] * (t_[1] - t_[0])  # 分界点
        plt.axvline(bound, linestyle='--', c="k", label="bound")  # 垂直于x轴的直线
        print(bound)

        #计算误差
        error = len(Y1[Y1 < bound]) + len(Y2[Y2 > bound])
        print("训练误差error = ", error)
    else:
        # 有两个交点时
        bound1 = min(t_) + point[0] * (t_[1] - t_[0])  # 较小的分界点
        bound2 = min(t_) + point[1] * (t_[1] - t_[0])  # 较大的分界点
        plt.axvline(bound1, linestyle='--', c="k", label="bound1")
        plt.axvline(bound2, linestyle='--', c="k", label="bound2")
        print(bound1, bound2)

        # 计算误差
        error = 0
        for i in range(len(Y2)):
            if bound1 < Y2[i] < bound2:
                error += 1
        for i in range(len(Y1)):
            if Y1[i] < bound1 or Y1[i] > bound2:
                error += 1
        print("训练误差error = ", error)

    plt.legend()
    plt.show()

density_est(X_2, X_3, w)


# (f)
density_est(X_2, X_3, np.array([[1.], [2.], [-1.5]]))
print("最优子空间中的训练误差为4, 非最优子空间中的训练误差为2，我觉得非常滑稽")

