# -- coding: UTF-8 --
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

data_iris = np.loadtxt("data_iris.csv", delimiter=",", usecols=[0,1,2,3])
X_1, X_2, X_3 = data_iris[:50, :], data_iris[50:100, :], data_iris[100:, :]
Y_dict = {1: "setosa", 2: "versicolor", 3: "virginica"}

def visual(X1, X2, X3, d_list):
    """对X1, X2, X3三类样本的d1, d2, d3三个维度可视化"""
    d1, d2, d3 = d_list
    ax = plt.subplot(111, projection="3d")
    ax.scatter(X1[:, d1], X1[:, d2], X1[:, d3], label="class1")
    ax.scatter(X2[:, d1], X2[:, d2], X2[:, d3], label="class2")
    ax.scatter(X3[:, d1], X3[:, d2], X3[:, d3], label="class3")
    ax.set_xlabel(str(d1) + "-axis", fontsize=16)
    ax.set_ylabel(str(d2) + "-axis", fontsize=16)
    ax.set_zlabel(str(d3) + "-axis", fontsize=16)
    plt.legend()
    plt.show()
# for d_list in [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]:
#     visual(X_1, X_2, X_3, d_list)


def build_dataset(X1, X2, seed0=None):
    X1, X2 = np.c_[X1, np.ones_like(X1[:,0])], np.c_[X2, np.ones_like(X1[:,0])]  # 增广一列
    X2 = - X2  # 将负类样本规范化

    random.seed(seed0)  # 设置随机数种子
    train_p_idx = random.sample(range(50), 25)  # 随机采样，p为X1, n为X2
    train_n_idx = random.sample(range(50), 25)
    random.seed(None)  # 清除随机数种子

    X1_train, X1_test = X1[train_p_idx, :], X1[[i for i in range(50) if i not in train_p_idx], :]
    X2_train, X2_test = X2[train_n_idx, :], X2[[i for i in range(50) if i not in train_n_idx], :]

    return np.concatenate([X1_train, X2_train],axis=0), \
           np.concatenate([X1_test, X2_test],axis=0)  # 返回训练集和测试集
#d_train, d_test = build_dataset(X_1, X_3, seed0=1)


def get_eta(k, method="constant", C=1):
    """
    生成当前迭代轮数下的学习率eta
    :param method: 通过何种方式来生成eta序列
    :param k: 当前迭代轮数（最小值为1）
    :param C: 仅当method=="constant"时生效，常数的学习率
    :return:
    """
    if method == "constant":
        eta = C
    elif method == "1/k":
        eta = C/k
    elif method == "k":
        eta = C*k
    else:
        eta = None
    return eta


def alg3_batch_perc(d_train, a_init, epsilon=1e-4, max_iter=50):
    """批处理感知器算法，以更新增量的模小于epslion为循环终止条件"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init # 初始化分类面法向量a

    i_iter = 0
    while True:
        bool_y = np.dot(d_train, a) <= np.zeros([n, 1])
        all_idx = np.array(range(n)).reshape(-1, 1)
        wrong_idx = all_idx[bool_y]  # 被错分样本点序号
        wrong_y = d_train[wrong_idx, :]  # 被错分的样本

        eta = get_eta(i_iter + 1)  # 学习率
        delta = eta * wrong_y.sum(axis=0).reshape(-1, 1)  # 更新增量

        if i_iter != 0:
            print("第", i_iter, "轮迭代后的错分点数量为：", wrong_idx.shape[0])
        if np.linalg.norm(delta) < epsilon:  # 更新增量太小就结束迭代
            break
        if i_iter == max_iter:
            print("经过", i_iter, "轮迭代后算法仍未收敛，错分点数量为：", wrong_idx.shape[0])
            break

        a += delta
        i_iter += 1
    return a


def alg4_single_perc(d_train, a_init, max_iter=20):
    """固定增量单样本感知器"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_pass, i_update = 0, 0  # i_pass: 样本集的循环次数, i_update: 对法向量的修正次数
    while True:
        num_wrong = 0  # 错分样本数量
        for i in range(n):
            if np.dot(d_train[i, :], a) <= 0:  # 分错了就更新法向量
                a += d_train[i, :].reshape(-1,1)
                i_update += 1
                print("第", i_update, "次更新中使用的样本序号为", i)
                num_wrong += 1
        i_pass += 1
        print("第", i_pass, "次pass中，法向量修正次数为", num_wrong, '\n')
        if num_wrong == 0:  # 样本集上无错分
            break
        if i_pass == max_iter:
            print("样本集经过了",i_pass, "pass之后，还是没收敛")
            break
    return a


def alg5_single_perc(d_train, a_init, b, max_iter=20):
    """带裕量的变增量感知器"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_pass, i_update = 0, 0  # i_pass: 样本集的循环次数, i_update: 对法向量的修正次数
    while True:
        num_wrong = 0  # 错分样本数量
        for i in range(n):
            if np.dot(d_train[i, :], a) <= b:  # 分错了就更新法向量
                eta = get_eta(i_pass+1, method="1/k")
                a += eta * d_train[i, :].reshape(-1,1)
                i_update += 1
                print("第", i_update, "次更新中使用的样本序号为", i)
                num_wrong += 1
        i_pass += 1
        print("第", i_pass, "次pass中，法向量修正次数为", num_wrong, '\n')
        if num_wrong == 0:  # 样本集上无错分
            break
        if i_pass == max_iter:
            print("样本集经过了",i_pass, "pass之后，还是没收敛")
            break
    return a


def alg6_batch_perc(d_train, a_init, max_iter=20):
    """批处理变增量感知器，以全部分对作为迭代终止条件"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_iter = 0
    while True:
        bool_y = np.dot(d_train, a) <= np.zeros([n, 1])
        all_idx = np.array(range(n)).reshape(-1, 1)
        wrong_idx = all_idx[bool_y]  # 被错分样本点序号
        wrong_y = d_train[wrong_idx, :]  # 被错分的样本

        if i_iter != 0:
            print("第", i_iter, "轮迭代后的错分点数量为：", wrong_idx.shape[0])
        if wrong_idx.shape[0] == 0:  # 没有错分样本点就结束迭代
            break
        if i_iter == max_iter:
            print("经过", i_iter , "轮迭代后算法仍未收敛，错分点数量为：", wrong_idx.shape[0])
            break

        eta = get_eta(i_iter + 1, method="1/k")  # 学习率
        a += eta * wrong_y.sum(axis=0).reshape(-1, 1)  # 更新法向量
        i_iter += 1
    # for i_iter in range(max_iter):
    #     bool_y = np.dot(d_train, a) <= np.zeros([n, 1])
    #     all_idx = np.array(range(n)).reshape(-1,1)
    #     wrong_idx = all_idx[bool_y]  # 被错分样本点序号
    #     wrong_y = d_train[wrong_idx,:]  # 被错分的样本
    #
    #     if i_iter != 0:
    #         print("第", i_iter, "轮迭代后的错分点数量为：", wrong_idx.shape[0])
    #
    #     if wrong_idx.shape[0] == 0:  # 没有错分样本点就结束迭代
    #         break
    #
    #     eta = get_eta(i_iter+1)  # 学习率
    #     a += eta * wrong_y.sum(axis=0).reshape(-1,1)  #更新法向量
    #
    # if i_iter == max_iter - 1 and wrong_idx.shape[0] != 0:
    #     print("经过", i_iter+1, "轮迭代后算法仍未收敛，错分点数量为：", wrong_idx.shape[0])
    return a


def alg8_batch_relax(d_train, a_init, b, max_iter=20):
    """批处理裕量松弛算法"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_iter = 0
    while True:
        bool_y = np.dot(d_train, a) <= b * np.ones([n, 1])
        all_idx = np.array(range(n)).reshape(-1, 1)
        wrong_idx = all_idx[bool_y]  # 被错分样本点序号
        wrong_y = d_train[wrong_idx, :]  # 被错分的样本

        if i_iter != 0:
            print("第", i_iter, "轮迭代后的裕量不足点数量为：", wrong_idx.shape[0])
        if wrong_idx.shape[0] == 0:  # 没有错分样本点就结束迭代
            break
        if i_iter == max_iter:
            print("经过", i_iter , "轮迭代后算法仍未收敛，裕量不足点数量为：", wrong_idx.shape[0])
            break

        eta = get_eta(i_iter+1, C=0.01)  # 学习率
        delta = 0
        for j in range(wrong_idx.shape[0]):
            delta += wrong_y[j,:].reshape(-1,1) * (b - np.dot(wrong_y[j,:], a)) / np.linalg.norm(wrong_y[j,:],2) ** 2

        a += eta * delta  # 更新法向量
        i_iter += 1
    return a


def alg9_single_relax(d_train, a_init, b, max_iter=20):
    """单样本裕量松弛算法"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_pass, i_update = 0, 0  # i_pass: 样本集的循环次数, i_update: 对法向量的修正次数
    while True:
        num_wrong = 0  # 错分样本数量
        for i in range(n):
            if np.dot(d_train[i, :], a) <= b:  # 分错了就更新法向量
                eta = get_eta(i_pass+1, C=2)
                a += eta * d_train[i,:].reshape(-1,1) * (b - np.dot(d_train[i, ], a)) / np.linalg.norm(d_train[i, ], 2) ** 2
                i_update += 1
                print("第", i_update, "次更新中使用的样本序号为", i)
                num_wrong += 1
        i_pass += 1
        print("第", i_pass, "次pass中，法向量修正次数为", num_wrong, '\n')
        if num_wrong == 0:  # 样本集上无错分
            break
        if i_pass == max_iter:
            print("样本集经过了",i_pass, "pass之后，还是没收敛")
            break
    return a


def alg10_lms(d_train, a_init, b, epsilon=1e-4, max_iter=20):
    """LMS算法"""
    n, d = d_train.shape  # 样本数量和维度（增广后）
    a = a_init  # 初始化分类面法向量a

    i_iter = 0
    i_all = 0
    while i_iter < max_iter:
        for i in range(n):
            i_all += 1
            eta = get_eta(i_all, method="1/k", C=0.1)
            delta = (b - np.dot(d_train[i,:], a)) * d_train[i,:].reshape(-1,1)
            if eta * np.linalg.norm(delta,2) < epsilon:  # 增量过小
                print(i_iter, i_all)
                return a
            print(eta * np.linalg.norm(delta,2))
            a += eta * delta
        i_iter += 1
        if i_iter == max_iter:
            print("经过", i_iter, "轮pass之后还是没收敛")
            print(i_iter, i_all)
            return a


def alg11_ho_kashyap(d_train, a_init, b_init=2, b_min = 1.5, max_iter=20):
    """Ho-Kashyap算法"""
    n, d = d_train.shape
    a, b = a_init, b_init * np.ones([n, 1])
    d_train_pinv = np.linalg.pinv(d_train)  # 广义逆

    i_iter = 0
    while i_iter < max_iter:
        e_ = np.dot(d_train, a) - b
        e_plus = 0.5 * (e_ + e_.__abs__())
        b += 2 * get_eta(i_iter+1, C=0.1) * e_plus
        #print(2 * get_eta(i_iter+1, method="1/k") * e_plus)
        a = np.dot(d_train_pinv, b)
        if (e_.__abs__() <= b_min).all():
            return a, b, i_iter, True
        i_iter += 1
    return False


def alg12_ho_kashyap(d_train, a_init, b_init=2, b_min = 1.5, max_iter=20):
    """修改的Ho-Kashyap算法"""
    n, d = d_train.shape
    a, b = a_init, b_init * np.ones([n, 1])
    d_train_pinv = np.linalg.pinv(d_train)  # 广义逆

    i_iter = 0
    while i_iter < max_iter:
        e_ = np.dot(d_train, a) - b
        e_plus = 0.5 * (e_ + e_.__abs__())
        b += 2 * get_eta(i_iter + 1, C=0.05) * (e_ + e_.__abs__())
        # print(2 * get_eta(i_iter+1, method="1/k") * e_plus)
        a = np.dot(d_train_pinv, b)
        if (e_.__abs__() <= b_min).all():
            return a, b, i_iter, True
        i_iter += 1
    return False



# ccc = alg12_ho_kashyap(d_train, np.ones([5,1]), max_iter=1000)
# #a = alg10_lms(d_train, np.ones([5,1]), 1, max_iter=100)
# a = alg8_batch_relax(d_train=d_train, a_init=np.ones([5,1]), b=1, max_iter=20)

accuacy = []
a_init = np.ones((5,1))
for i_seed in range(100):
    d_train, d_test = build_dataset(X_1, X_3, seed0=i_seed)
    a, _, __, ___ = alg11_ho_kashyap(d_train, a_init, b_init=2, b_min = 5, max_iter=100000)
    print(a)
    y_pred = d_test.dot(a)
    #print(y_pred)
    accuacy.append(len(y_pred[y_pred > 0]) / len(d_test))
print(np.mean(accuacy), np.var(accuacy))
