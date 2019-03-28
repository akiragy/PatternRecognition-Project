# -- coding: UTF-8 --
import numpy as np


class BaseErrorCorrectingAlg(object):
    """误差校正方法：感知器算法（alg3-6）和松弛算法（alg8-9）的基类"""

    def __init__(self, loss, margin, batch_size, max_iter, learning_rate, eta0, penalty, alpha):
        self.loss = loss  # 损失函数，包括感知器准则函数和二次松弛准则函数
        self.margin = margin  # 边界裕量
        self.batch_size = batch_size  # 一批样本的数量，batch = -1意味着在一轮迭代中使用全部样本
        self.max_iter = max_iter  # 对整个数据集的最大pass数
        self.learning_rate = learning_rate  # 学习率类型
        self.eta0 = eta0  # 初始学习率
        self.penalty = penalty  # 正则化项
        self.alpha = alpha  # 正则化系数

    def _get_eta(self, k):
        """给出第k步迭代的学习率。目前只给出了最简单的三种"""
        if self.learning_rate == "constant":  # 恒定学习率
            eta = self.eta0
        elif self.learning_rate == "1/k":  # 反比例衰减的学习率
            eta = self.eta0 / k
        elif self.learning_rate == "k":  # 线性增长的学习率
            eta = self.eta0 * k
        else:
            raise ValueError("目前并没有这种学习率...")
        return eta

    def _gradient_descent_for_penalty(self, n_):
        """对正则化项进行梯度下降"""
        if self.penalty == "none":
            pass
        elif self.penalty == "l2":
            self.coef_ -= self._get_eta(self.i_iter + 1) * self.alpha * n_ * self.coef_

    def _prepare_fit(self, X, y):
        """格式检查，初始化待估参数。健壮性代码可以全部放在此函数中"""
        if len(y.shape) != 1:
            raise ValueError("请保证y只有一个维度...")
        if set(y.tolist()) != set([-1.,1.]):
            raise ValueError("请保证y的值为-1和1...")
        n, d = X.shape
        if self.batch_size < 0:
            self.batch_size = n
        #self.coef_ = np.random.normal(0, 0.1, [d,])  # 初始化权向量
        self.coef_ = np.ones([d,])
        self.intercept_ = 0.  # 初始化截距
        self.i_iter = 0  # 实际迭代次数

    def _fit(self, X, y):
        """训练流程，其中的_gradient_descent方法需要不同的准则函数各自实现"""
        _num_continu_iter_no_wrong = 0  # 用于记录多少轮未出现错分样本
        while self.i_iter < self.max_iter:
            np.random.seed(None)

            # 此处应当添加对数据集进行完全pass的功能，可以使用队列作为buffer

            batch_idx = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
            wrong_idx = batch_idx[y[batch_idx] * (X[batch_idx, :].dot(self.coef_)+self.intercept_) <= self.margin]

            if len(wrong_idx) != 0:
                self._gradient_descent(X, y, wrong_idx)  # 核心步骤：对目标函数进行梯度下降
                self._gradient_descent_for_penalty(len(wrong_idx))  # 对正则化项进行梯度下降
                _num_continu_iter_no_wrong = 0
            else:
                _num_continu_iter_no_wrong += 1  # 本轮iter中的样本全部分对

            if _num_continu_iter_no_wrong >= 2 * X.shape[0] / self.batch_size:  # 连续分对iter轮数大于阈值则近似认为已经收敛
                self.i_iter -= _num_continu_iter_no_wrong - 1  # 减去用于检验收敛的迭代次数
                break
            self.i_iter += 1

    def fit(self, X, y):
        """训练"""
        self._prepare_fit(X, y)
        self._fit(X, y)

    def predict(self, X):
        """测试"""
        y = X.dot(self.coef_) + self.intercept_
        return [1 if y[i] >= 0 else -1 for i in range(len(y))]


class Perceptron(BaseErrorCorrectingAlg):
    """感知器算法"""

    def __init__(self, margin=0, batch_size=-1, max_iter=1000, learning_rate="constant", eta0=0.01,
                 penalty="none", alpha = 0):
        super(Perceptron, self).__init__(loss="perceptron", margin=margin,
                                         batch_size=batch_size, max_iter=max_iter,
                                         learning_rate=learning_rate, eta0=eta0,
                                         penalty=penalty, alpha=alpha)

    def _gradient_descent(self, X, y, sample_idx):
        """对感知器准则函数进行梯度下降"""
        self.coef_ += self._get_eta(self.i_iter + 1) * np.diag(y[sample_idx]).dot(X[sample_idx, :]).sum(axis=0)
        self.intercept_ += self._get_eta(self.i_iter + 1) * y[sample_idx].sum()


class RelaxationPro(BaseErrorCorrectingAlg):
    """松弛算法"""

    def __init__(self, margin=1, batch_size=-1, max_iter=1000, learning_rate="constant", eta0=0.01,
                 penalty="none", alpha=0):
        super(RelaxationPro, self).__init__(loss="relaxtionpro", margin=margin,
                                            batch_size=batch_size, max_iter=max_iter,
                                            learning_rate=learning_rate, eta0=eta0,
                                            penalty=penalty, alpha=alpha)

    def _gradient_descent(self, X, y, sample_idx):
        """对松弛算法准则函数进行梯度下降"""
        for idx_ in sample_idx:
            grad_common = y[idx_] * (y[idx_]*(X[idx_,:].dot(self.coef_)+self.intercept_) - self.margin) / np.linalg.norm(X[idx_,:]) ** 2
            self.coef_ -= self._get_eta(self.i_iter + 1) * grad_common * X[idx_,:]
            self.intercept_ -= self._get_eta(self.i_iter + 1) * grad_common


class SVM(BaseErrorCorrectingAlg):
    """使用SGD优化的SVM"""
    def __init__(self, batch_size=-1, max_iter=1000, learning_rate="constant", eta0=0.01,
                 penalty="l2", alpha=0.001):
        super(SVM, self).__init__(loss="hinge", margin=1,
                                  batch_size=batch_size, max_iter=max_iter,
                                  learning_rate=learning_rate, eta0=eta0,
                                  penalty=penalty, alpha=alpha)

    def _gradient_descent(self, X, y, sample_idx):
        """对hinge loss进行梯度下降"""
        self.coef_ += self._get_eta(self.i_iter + 1) * np.diag(y[sample_idx]).dot(X[sample_idx, :]).sum(axis=0)
        self.intercept_ += self._get_eta(self.i_iter + 1) * y[sample_idx].sum()
