# -- coding: UTF-8 --
import numpy as np

DEFAULT_MAX_ITER = 1000  # 默认最大迭代次数
DEFAULT_TOL = 1e-4  # 默认收敛条件
DEFAULT_LEARNING_RATE = "constant"  #  默认学习率选择方式
DEFAULT_ETA0 = 0.01  # 默认初始学习率
DEFAULT_ALPHA = 1  # 默认正则化系数

def gradient_descent_(X, y, is_batch, eta, loss, margin, penalty, alpha, tol, coef_, intercept_):
    """对不同的loss和penalty进行一次梯度下降"""

    # 计算经验损失（loss function）的梯度
    if loss == "hinge":
        wrong_idx = np.array(range(len(y)))[y * (X.dot(coef_) + intercept_) <= margin]  # 错分样本
        coef_loss = -eta * np.diag(y[wrong_idx]).dot(X[wrong_idx, :]).sum(axis=0)
        intercept_loss = -eta * y[wrong_idx].sum()
    elif loss == "log":
        pass
    elif loss == "mse":
        pass
    else:
        raise ValueError("不支持此种损失函数...")

    # 根据loss function的梯度判断是否收敛（仅对fit方法生效，fit_batch算法不需要判断）
    if (not is_batch) and np.linalg.norm(coef_loss) <= tol:
        return coef_, intercept_, True  # 已收敛，不需要进行梯度下降

    # 计算正则化项（penalty function）的梯度
    if penalty == "none":
        coef_penalty = 0
    elif penalty == "l2":
        coef_penalty = eta * alpha * coef_
    elif penalty == "l1":
        pass
    else:
        raise ValueError("不支持此种正则化项...")

    return coef_ - coef_loss - coef_penalty, intercept_ - intercept_loss, False  # 梯度下降


class BaseSGD(object):
    """使用随机梯度下降（SGD）的分类器/回归器基类"""

    def __init__(self, loss, margin, max_iter, tol, learning_rate, eta0, penalty, alpha):
        self.loss = loss  # 损失函数，包括感知器准则函数和二次松弛准则函数
        self.margin = margin  # hinge loss的margin
        self.max_iter = max_iter  # 对整个数据集的最大pass数
        self.tol = tol  # 迭代终止条件
        self.learning_rate = learning_rate  # 学习率类型
        self.eta0 = eta0  # 初始学习率
        self.penalty = penalty  # 正则化项
        self.alpha = alpha  # 正则化系数

        self.coef_ = None
        self.intercept_ = None

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

    def _prepare_fit(self, X, y, is_batch):
        """格式检查，初始化待估参数。健壮性代码可以全部放在此函数中"""

        if len(y.shape) != 1:
            raise ValueError("请保证y只有一个维度...")
        if set(y.tolist()) != set([-1.,1.]):
            raise ValueError("请保证y的值为-1和1...")

        self.i_iter = 0  # 实际迭代次数
        n, d = X.shape
        if is_batch:
            if self.coef_ is None or self.intercept_ is None:
                self.coef_ = np.ones([d,])  # 对于fit_batch方法，仅初次调用需要初始化参数
                self.intercept_ = 0.
        else:
            self.coef_ = np.ones([d, ])  # 对于fit方法，总是初始化参数
            self.intercept_ = 0.

    def _fit(self, X, y, max_iter, is_batch):
        """通拟合数据集"""

        while self.i_iter < max_iter:
            eta = self._get_eta(self.i_iter + 1)
            self.coef_, self.intercept_, is_convergence = gradient_descent_(
                X, y, is_batch, eta, self.loss, self.margin, self.penalty, self.alpha, self.tol, self.coef_, self.intercept_
            )

            if is_convergence:  # 对于fit_batch方法，此条件永不满足
                break
            self.i_iter += 1

    def fit(self, X, y):
        self._prepare_fit(X, y, is_batch=False)
        self._fit(X, y, max_iter=self.max_iter, is_batch=False)

    def fit_batch(self, X, y):
        self._prepare_fit(X, y, is_batch=True)
        self._fit(X, y, max_iter=1, is_batch=True)  # 对于当前batch只迭代一轮


class SGDClassifier(BaseSGD):
    """使用随机梯度下降（SGD）的分类器"""

    def __init__(self, loss="hinge", margin=1,
                 max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL,
                 learning_rate=DEFAULT_LEARNING_RATE, eta0=DEFAULT_ETA0,
                 penalty="l2", alpha=DEFAULT_ALPHA):

        super(SGDClassifier, self).__init__(loss=loss, margin=margin,
                                            max_iter=max_iter, tol=tol,
                                            learning_rate=learning_rate, eta0=eta0,
                                            penalty=penalty, alpha=alpha)

    def predict(self, X):
        """预测"""
        y = X.dot(self.coef_) + self.intercept_
        return [1 if y[i] >= 0 else -1 for i in range(len(y))]


class Perceptron(SGDClassifier):
    """感知器"""

    def __init(self, margin=0,
               max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL,
               learning_rate=DEFAULT_LEARNING_RATE, eta0=DEFAULT_ETA0,
               penalty="none", alpha=DEFAULT_ALPHA):

        super(Perceptron ,self).__init__(loss="hinge", margin=margin,
                                         max_iter=max_iter, tol=tol,
                                         learning_rate=learning_rate, eta0=eta0,
                                         penalty=penalty, alpha=alpha)

class LinearSVC(SGDClassifier):
    """线性核的支持向量机"""

    def __init__(self,
                 max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL,
                 learning_rate=DEFAULT_LEARNING_RATE, eta0=DEFAULT_ETA0,
                 penalty="l2", alpha=DEFAULT_ALPHA):

        super(LinearSVC, self).__init__(loss="hinge", margin=1,
                                        max_iter=max_iter, tol=tol,
                                        learning_rate=learning_rate, eta0=eta0,
                                        penalty=penalty, alpha=alpha)










