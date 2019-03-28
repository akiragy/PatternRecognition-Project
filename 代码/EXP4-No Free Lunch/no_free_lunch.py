import numpy as np
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

def f_min(a, b):
    return True if a > b else False

def f_max(a, b):
    return True if a < b else False

class DoCV(object):

    def __init__(self):
        pass

    def create_datasets(self):
        #np.random.seed(None)
        self.dtest = np.random.rand(20, 3)
        self.dtest = np.random.laplace(size=[20,3])
        self.dtest[:10, 2], self.dtest[10:, 2] = 0, 1  # 第三列是标记

        self.dlabeled = np.random.rand(100, 3)
        self.dlabeled = np.random.laplace(size=[100,3])
        self.dlabeled[:50, 2], self.dlabeled[50:, 2] = 0, 1

    def do_cv(self, f_extre, skf_seed, n_splits=10):
        X = self.dlabeled[:, 0:2]
        y = self.dlabeled[:, 2]
        skf = StratifiedKFold(n_splits=n_splits, random_state=skf_seed, shuffle=True)

        k_cv = 1
        cvs = {}
        while k_cv <= y.shape[0] * (1-1/n_splits)-1:

            num_correct = 0
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                clf = neighbors.KNeighborsClassifier(n_neighbors=k_cv)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                num_correct += len([1 for i in range(len(y_val)) if y_pred[i] == y_val[i]])
            cvs[k_cv] = num_correct

            if k_cv > 1:
                if f_extre(num_correct, cvs[k_cv-2]):  # 第一个极值点
                    #print(cvs)
                    return k_cv - 2
            k_cv += 2
        return k_cv

    def test(self):
        self.create_datasets()
        skf_seed = 2018  # 需要保证最小值和最大值是对同一个数据集顺序找的
        self.k_min = self.do_cv(f_extre=f_min, skf_seed=skf_seed)  # 误差极小值的k
        self.k_max = self.do_cv(f_extre=f_max, skf_seed=skf_seed)  # 误差极大值的k

        errors = []
        X, y = self.dtest[:, 0:2], self.dtest[:, 2]
        for key, value in {"cv极小值的k":self.k_min, "cv极大值的k":self.k_max}.items():
            clf = neighbors.KNeighborsClassifier(n_neighbors=value)
            clf.fit(self.dlabeled[:, 0:2], self.dlabeled[:, 2])
            y_pred = clf.predict(X)
            error = len([1 for i in range(len(y)) if y_pred[i] == y[i]]) / len(y)
            errors.append(error)
            #print('\t'+key+"的错误率为：",  error)
        return np.array(errors), self.k_min, self.k_max


if __name__ == "__main__":
    fcv = DoCV()

    print("\t{:^}\t{:^}\t{:^}\t{:^}\t{:^}".format("实验序号", "极小值k的测试误差", "极大值k的测试误差", "极小值k", "极大值k"))
    errors = np.zeros([5,2])
    np.random.seed(0)
    for i in range(5):
        errors[i, :], k_min, k_max = fcv.test()
        print("\t{:^10}\t{:^10}\t{:^10}\t{:^10}\t{:^10}".format(i, errors[i,0], errors[i,1], k_min, k_max))
    print("\t{:^10}\t{:7.2f}\t{:11.2f}".format("平均", np.mean(errors, axis=0)[0], np.mean(errors, axis=0)[1]))


