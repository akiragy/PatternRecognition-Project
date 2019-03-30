import numpy as np
from scipy.linalg import eigh


class PCA(object):

    def __init__(self, n_components=None, solver="svd", dual=None):
        self.n_components = n_components
        self.dual = dual
        self.solver = solver

    def fit(self, X, y=None):

        self.mean_ = X.mean(axis=0)
        X_centralized = X - self.mean_  # 中心化后的样本
        n_samples, n_features = X.shape

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        self.n_components = min(self.n_components, n_samples, n_features)

        if self.solver == "svd":
            u, s, vh = np.linalg.svd(X_centralized, full_matrices=False)
            self.eigenvalues_ = s[:self.n_components] ** 2
            self.eigenvalues_rate = np.sum(s[:self.n_components] ** 2) / np.sum(s ** 2)
            self.components_ = vh.transpose()[:, :self.n_components]
        elif self.solver == "evd" and self.dual == False:
            S = np.dot(X_centralized.transpose(), X_centralized)  # 协方差矩阵
            e, u = np.linalg.eig(S)
            u = u[:, np.argsort(-e)]  #  排序
            e = -np.sort(-e)
            self.eigenvalues_ = e[:self.n_components]
            self.eigenvalues_rate = np.sum(e[:self.n_components]) / np.sum(e)
            self.components_ = u[:, :self.n_components]
            self.components_ /= np.linalg.norm(self.components_)  # 标准化
        elif self.solver == "evd" and self.dual == True:
            # K = np.dot(X_centralized, X_centralized.transpose())  # gram矩阵
            # e, a = np.linalg.eig(K)
            # a = a[:, np.argsort(-e)]  # 排序
            # e =  -np.sort(-e)
            # self.eigenvalues_ = e[:self.n_components]
            # self.eigenvalues_rate = np.sum(e[:self.n_components]) / np.sum(e)
            # self.components_ = np.dot(X_centralized.transpose(), a[:, :self.n_components])
            # self.components_ /= np.linalg.norm(self.components_)  # 标准化
            K = np.dot(X_centralized, X_centralized.transpose())  # gram矩阵
            e, a = np.linalg.eig(K)
            a = a[:, np.argsort(-e)]  # 排序
            e = -np.sort(-e)
            self.eigenvalues_ = e[:self.n_components]
            self.eigenvalues_rate = np.sum(e[:self.n_components]) / np.sum(e)
            self.components_ = X_centralized.transpose().dot(a[:, :self.n_components]).dot(np.diag(np.power(e[:self.n_components], -0.5)))
            self.components_ /= np.linalg.norm(self.components_)  # 标准化
        else:
            raise ValueError("solver和dual选择不正确")

    def transform(self, X):
        return np.dot(X - self.mean_, self.components_)

class DPDR(object):

    def fit(self, X, y=None):

        n_samples, n_features = X.shape
        if n_samples >= n_features:
            raise  ValueError("只有当特征数量大于样本数量时才能使用")

        self.coef_, r = np.linalg.qr(X.transpose(), mode="reduced")

    def transform(self, X):
        return np.dot(X, self.coef_)

class LDA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)  # 类别数量

        X_dict = {}
        for i in range(n_samples):
            if y[i] not in X_dict:
                X_dict[y[i]] = X[i, :]
            else:
                X_dict[y[i]] = np.row_stack((X_dict[y[i]], X[i, :]))

        self.xbar_ = X.mean(axis=0)  # 总体均值
        self.means_ = np.zeros((n_classes, n_features))  # 每一类的均值
        self.Sw_ = np.zeros((n_features, n_features))  # 类内散布矩阵
        self.Sb_ = np.zeros((n_features, n_features))  # 类间散布矩阵
        for i in range(n_classes):
            self.means_[i, :] = X_dict[classes[i]].mean(axis=0)
            self.Sw_ += (X_dict[classes[i]] - self.means_[i, :]).transpose().dot(X_dict[classes[i]] - self.means_[i, :])

            self.Sb_ += X_dict[classes[i]].shape[0] * (self.means_[[i], :] - self.xbar_).transpose().dot(self.means_[[i], :] - self.xbar_)

        e, u = np.linalg.eig(np.linalg.inv(self.Sw_ + 0 * np.eye(n_features)).dot(self.Sb_))
        #e, u = eigh(self.Sb_, self.Sw_)
        e, u = np.real(e), np.real(u)
        u = u[:, np.argsort(-e)]  # 排序
        e = -np.sort(-e)
        self.coef_ = u[:self.n_components].T

    def transform(self, X):
        return X.dot(self.coef_)

