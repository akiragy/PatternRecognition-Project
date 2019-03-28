import numpy as np

class KMeans(object):

    def __init__(self, n_clusters, init="random", dist_metric="Euclidean", tol=1e-4, max_iter=500, **kwargs_func_dist):
        """
        KMeans
        :param n_clusters:  聚类中心个数
        :param init:  初始化聚类中心，可以是numpy数组或者"random"
        :param distance:  距离度量，可以传入函数或欧氏距离"Euclidean"
        :param tol:  聚类中心变化小于此值时判为收敛
        :param max_iter:  最大迭代次数
        :param **kwargs_func_dist:  自定义距离函数的参数
        """
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.kwargs_func_dist = kwargs_func_dist
        self.n_iter = None  # 实际迭代次数

        if hasattr(dist_metric, "__call__"):
            self.func_dist = dist_metric
        elif dist_metric == "Euclidean":
            self.func_dist = lambda x, y: np.sqrt(np.linalg.norm(x-y) ** 2)  # 这个平方再开方写得太TM优雅了


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

            # step2 为样本分配聚簇
            labels_old = self.labels_.copy()
            for i in range(n_samples):
                distance_n = distance_nk[i, :]
                self.labels_[i] = np.where(distance_n == np.min(distance_n))[0][0]  # 若到两个中心的距离相等，则分到序号小的

            # step3 更新聚类中心
            cluster_centers_old = self.cluster_centers_.copy()
            self.cluster_centers_ = np.zeros((n_clusters, n_features))
            n_samples_per_cluster = np.zeros(n_clusters)  # 每个聚簇含有的样本数量
            for i in range(n_samples):
                # for j in range(n_clusters):
                self.cluster_centers_[int(self.labels_[i]), :] += X[i, :]
                n_samples_per_cluster[int(self.labels_[i])] += 1
            for j in range(n_clusters):
                if n_samples_per_cluster[j] != 0:
                    self.cluster_centers_[j, :] /= n_samples_per_cluster[j]
                else:
                    self.cluster_centers_[j, :] = cluster_centers_old[j, :]

            # 判断收敛
            if np.linalg.norm(cluster_centers_old - self.cluster_centers_) < self.tol:
                return


    def predict(self, X, y=None):
        """预测X中的样本属于哪一个聚簇"""
        pass


    def transform(self, X, y=None):
        """计算X中样本到每一个聚簇的距离"""