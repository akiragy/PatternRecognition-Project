# -- coding: UTF-8 --
import numpy as np

data_all = np.loadtxt("data.csv", delimiter=',')
data_use = data_all[:10, :]

# (a)
def MLE_1D(data, dim):
    """一维情况"""
    assert dim in [0,1,2]
    X = data[:,[dim]]
    mu = np.sum(X) / np.shape(X)[0]
    sigma = np.dot(np.transpose(X - mu), X - mu) / np.shape(X)[0]
    return mu, sigma

print("一维情况下：")
for i in range(3):
    mu, sigma = MLE_1D(data_use, i)
    print("for dimension", i+1, ": mu = ", mu, ", sigma = ", sigma)


# (b)
def MLE_2D(data, dim1, dim2):
    """二维情况"""
    assert dim1 in [0,1,2] and dim2 in [0,1,2]
    X = data[:,[dim1, dim2]]
    mu = np.sum(X, axis=0, keepdims=True) / np.shape(X)[0]
    sigma = np.dot(np.transpose(X - mu), X - mu) / np.shape(X)[0]
    return mu, sigma

print("\n二维情况下：")
for i in range(3):
    mu, sigma = MLE_2D(data_use, i, (i+1)%3)
    print("for dimension", i+1, "and", (i+1)%3+1, " : mu = ", mu, ", sigma = ", sigma)


# (c)
def MLE_3D(data):
    """三维情况"""
    X = data.copy()
    mu = np.sum(X, axis=0, keepdims=True) / np.shape(X)[0]
    sigma = np.dot(np.transpose(X - mu), X - mu) / np.shape(X)[0]
    return mu, sigma

print("\n三维情况下：")
mu, sigma = MLE_3D(data_use)
print("mu = ", mu, ", sigma = ", sigma)


# (d)
def MLE_3D_sep(data):
    """三维独立情况"""
    X = data.copy()
    mu1 = np.sum(X[:, 0], axis=0, keepdims=True) / np.shape(X)[0]
    mu2 = np.sum(X[:, 1], axis=0, keepdims=True) / np.shape(X)[0]
    mu3 = np.sum(X[:, 2], axis=0, keepdims=True) / np.shape(X)[0]
    sigma1 = np.dot(np.transpose(X[:, 0] - mu1), X[:, 0] - mu1) / np.shape(X)[0]
    sigma2 = np.dot(np.transpose(X[:, 1] - mu2), X[:, 1] - mu2) / np.shape(X)[0]
    sigma3 = np.dot(np.transpose(X[:, 2] - mu3), X[:, 2] - mu3) / np.shape(X)[0]
    print("mu = ", mu1, mu2, mu3, ", sigma = ", sigma1, sigma2, sigma3)

print("\n可分离情况下：")
MLE_3D_sep(data_use)