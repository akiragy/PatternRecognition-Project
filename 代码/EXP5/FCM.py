import numpy as np

class FCM(object):

    def __init__(self, n_clusters, b = 2, init="random", dist_metric="Euclidean", tol=1e-4, max_iter=500, **kwargs_func_dist):
        """
        模糊K均值
        :param n_clusters:  聚类中心个数
        :param b:
        :param init:  初始化聚类中心，可以是numpy数组或者"random"
        :param dist_metric:  距离度量，可以传入函数或欧氏距离"Euclidean"
        :param tol:  聚类中心变化小于此值时判为收敛
        :param max_iter:  最大迭代次数
        :param kwargs_func_dist:  自定义距离函数的参数
        """
        self.b = b
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.kwargs_func_dist = kwargs_func_dist
        self.n_iter = None  # 实际迭代次数

        if hasattr(dist_metric, "__call__"):
            self.func_dist = dist_metric
        elif dist_metric == "Euclidean":
            self.func_dist = lambda x, y: np.linalg.norm(x-y)


    def fit(self, X, y=None):

        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        # 初始化
        if isinstance(self.init, np.ndarray):
            if self.init.shape == (self.n_clusters, n_features):
                self.cluster_centers_ = self.init  # 使用给定值初始化聚类中心
            else:
                raise ValueError("所给的聚类中心初值格式不正确...")
        elif self.init == "random":
            self.cluster_centers_ = np.random.uniform(X.min(), X.max(), (n_clusters, n_features))  # 随机初始化聚类中心

        self.fuzzy_membership_ = np.zeros((n_samples, n_clusters))  # 每个样本对每个类别的隶属度[n_samples, n_clusters]
        self.labels_ = np.zeros(n_samples)  # 每个样本所属聚簇 [n_samples, ]

        # 迭代
        distance_nk = np.zeros((n_samples, n_clusters))  # 记录每个样本到每个聚类中心的距离
        self.n_iter = 0
        while self.n_iter < self.max_iter:

            self.n_iter += 1
            # step1 计算样本到聚类中心的距离
            for i in range(n_samples):
                for j in range(n_clusters):
                    distance_nk[i, j] = self.func_dist(X[i, :], self.cluster_centers_[j, :], **self.kwargs_func_dist)

            # step2 计算样本属于聚簇的后验概率，即隶属度
            for i in range(n_samples):
                for j in range(n_clusters):
                    self.fuzzy_membership_[i, j] = (1 / distance_nk[i, j]) ** (2 / (self.b-1))  # 会发生下溢出，继而噪声均值的上溢出
            self.fuzzy_membership_ /= self.fuzzy_membership_.sum(axis=1, keepdims=True)  # 归一化

            # step3 更新聚类中心
            cluster_centers_old = self.cluster_centers_.copy()
            self.cluster_centers_ = np.zeros((n_clusters, n_features))
            norm_coef_ = np.zeros(n_clusters)  # 归一化
            for i in range(n_samples):
                for j in range(n_clusters):
                    norm_coef_[j] += self.fuzzy_membership_[i, j] ** self.b
                    self.cluster_centers_[j, :] += self.fuzzy_membership_[i, j] ** self.b * X[i, :]

            for j in range(n_clusters):
                if norm_coef_[j] > 1e-8:
                    self.cluster_centers_[j, :] /= norm_coef_[j]
                else:
                    raise ValueError("计算均值时出现数值上溢出，请尝试上调b的值")

            # 为样本分配聚簇
            for i in range(n_samples):
                fuzzy_n = self.fuzzy_membership_[i, :]
                self.labels_[i] = np.where(fuzzy_n == np.max(fuzzy_n))[0][0]

            # 判断收敛
            if np.linalg.norm(cluster_centers_old - self.cluster_centers_) < self.tol:
                return