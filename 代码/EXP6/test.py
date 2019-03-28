import numpy as np
from PIL import Image
from dimension_reduction import PCA, DPDR, LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA as sk_PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sk_lda
import time

n_classes = 40  # 类别数量
n_samples = 10  # 每个类别的样本数量
n_train = 5  # 每个类别的训练样本数量
h_ = 112  # 图像高度(矩阵行数)
w_ = 92  # 图像宽度(矩阵列数)
img_path = "att_faces/"

data = np.zeros((n_classes, n_samples, w_*h_))  # 储存所有图像
for i in range(n_classes):
    for j in range(n_samples):
        im = Image.open(img_path + "s" + str(i+1) + "/" + str(j+1) + ".pgm")
        data[i, j, :] = np.array(im).reshape(-1) / 255

dtrain, dtest = data[:, :n_train, :], data[:, n_train:, :]

# 划分训练集和测试集
X_train, X_test = dtrain.reshape(-1, w_*h_), dtest.reshape(-1, w_*h_) # [样本数量, 样本维度]
y_train, y_test = np.linspace(1, n_classes, n_classes).repeat(n_train), \
                  np.linspace(1, n_classes, n_classes).repeat(n_samples-n_train)

knn = KNeighborsClassifier(n_neighbors=1)

# # 第一题：PCA降维后KNN分类
# pca = PCA(n_components=80, solver="evd", dual=False)
# #pca = sk_PCA(n_components=10)  # 用sklearn的PCA检验
# t_start = time.time()
# pca.fit(X_train)
# t_end = time.time()
# t_ = t_end- t_start
# #print("特征值：", pca.eigenvalues_, "\n特征值占比：", pca.eigenvalues_rate,  '\n特征向量：', pca.components_)
# X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)
#
#knn.fit(X_train_pca, y_test)
# y_pred_pca = knn.predict(X_test_pca)
# acuracy_pca = len([1 for i in range(len(y_test)) if y_pred_pca[i] == y_test[i]]) / len(y_test)


# # 第二题：MDA降维后KNN分类
# lda = sk_lda(n_components=20)
# t_start = time.time()
# lda.fit(X_train, y_train)
# t_end = time.time()
# t_ = t_end- t_start
# X_train_lda, X_test_lda = lda.transform(X_train), lda.transform(X_test)
#
# knn.fit(X_train_lda, y_test)
# y_pred_lda = knn.predict(X_test_lda)
# acuracy_lda = len([1 for i in range(len(y_test)) if y_pred_lda[i] == y_test[i]]) / len(y_test)


# # 第三题：DPDR降维后KNN分类
# dpdr = DPDR()
# t_start = time.time()
# dpdr.fit(X_train)
# t_end = time.time()
# t_ = t_end - t_start
# X_train_dpdr, X_test_dpdr = dpdr.transform(X_train), dpdr.transform(X_test)
#
# knn.fit(X_train_dpdr, y_test)
# y_pred_dpdr = knn.predict(X_test_dpdr)
# acuracy_dpdr = len([1 for i in range(len(y_test)) if y_pred_dpdr[i] == y_test[i]]) / len(y_test)


# 第四题：外推能力
pca = PCA(n_components=80)
pca.fit(X_train)
X_test_restru_1 = np.dot(pca.transform(X_test + pca.mean_), pca.components_.transpose())
error_1 = np.linalg.norm(X_test - X_test_restru_1) ** 2 / 200

pca.fit(np.r_[X_train, X_test])
X_test_restru_2 = np.dot(pca.transform(X_test + pca.mean_), pca.components_.transpose())
error_2 = np.linalg.norm(X_test - X_test_restru_2) ** 2 / 200

dpdr = DPDR()
dpdr.fit(X_train)
X_test_restru_3 = np.dot(dpdr.transform(X_test), dpdr.coef_.transpose())
error_3 = np.linalg.norm(X_test - X_test_restru_3) ** 2 / 200

dpdr = DPDR()
dpdr.fit(np.r_[X_train, X_test])
X_test_restru_4 = np.dot(dpdr.transform(X_test), dpdr.coef_.transpose())
error_4 = np.linalg.norm(X_test - X_test_restru_4) ** 2 / 200


'''
lda = LDA(n_components=39)
t_start = time.time()
lda.fit(X_train, y_train)
t_end = time.time()
t_ = t_end - t_start
X_train_lda, X_test_lda = lda.transform(X_train), lda.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_lda, y_train)
y_pred_lda = knn.predict(X_test_lda)
acuracy_lda = len([1 for i in range(len(y_test)) if y_pred_lda[i] == y_test[i]]) / len(y_test)
'''

