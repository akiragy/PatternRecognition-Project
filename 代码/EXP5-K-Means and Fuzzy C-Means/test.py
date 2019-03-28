import numpy as np
from KMeans import KMeans
from FCM import FCM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("data.csv", delimiter=",")
data_iris = np.loadtxt("data_iris.csv", delimiter=",", usecols=[0,1,2,3])

# 聚类中心的初始值
cents = {}
cents[1] = np.array([[1, 1, 1], [-1, 1, -1]])
cents[2] = np.array([[0, 0, 0], [1, 1, -1]])
cents[3] = np.array([[0, 0, 0], [1, 1, 1], [-1, 0, 2]])
cents[4] = np.array([[-0.1, 0, 0.1], [0, -0.1, 0.1], [-0.1, -0.1, 0.1]])


# 方法1：通过向KMeans对象传递**kwargs。若KMeans的构造函数中有两个需要这样传参的函数会引起混淆
def my_dist1(x, y, dist_beta):
    """自定义距离函数"""
    return np.sqrt(1 - np.exp(-dist_beta * np.linalg.norm(x-y) ** 2))


# 方法2：通过装饰器，很麻烦
def custom_dist(**kwargs):
    """装饰器，用于向自定义距离函数传递参数"""
    def decorator(func):
        def wrapper(*args):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@custom_dist(dist_beta=0.001)
def my_dist2(x, y, dist_beta):
    return np.sqrt(1 - np.exp(-dist_beta * np.linalg.norm(x - y) ** 2))


# 方法3：通过默认参数，还凑合
def my_dist3(x, y, dist_beta=1):
    return np.sqrt(1 - np.exp(-dist_beta * np.linalg.norm(x - y) ** 2))

# Kmeans和FCM
ms = [">", "<", "s"]
for q in [1]:
    dist_metric = my_dist3#"Euclidean"
    km = KMeans(n_clusters=len(cents[q]), init=cents[q], dist_metric=dist_metric)
    km.fit(data)

    ax = plt.subplot(2,4,q, projection='3d')
    for i, label in enumerate(np.unique(km.labels_)):
        data_part = data[km.labels_ == label]
        ax.scatter(data_part[:, 0], data_part[:, 1], data_part[:, 2])#, label="cluster"+str(i+1))  # 画投影前的点
        ax.scatter(km.cluster_centers_[i,0], km.cluster_centers_[i,1], km.cluster_centers_[i,2], marker=ms[i], c="k", label="center"+str(i+1))
        plt.legend(loc="upper right")
        plt.title("K-Means-Q"+str(q))

    fcm = FCM(n_clusters=len(cents[q]), init=cents[q], b=2, dist_metric=dist_metric)
    fcm.fit(data)

    ax = plt.subplot(2,4,4+q, projection='3d')
    for i, label in enumerate(np.unique(fcm.labels_)):
        data_part = data[fcm.labels_ == label]
        ax.scatter(data_part[:, 0], data_part[:, 1], data_part[:, 2])#, label="cluster"+str(i+1))  # 画投影前的点
        ax.scatter(fcm.cluster_centers_[i,0], fcm.cluster_centers_[i,1], fcm.cluster_centers_[i,2], marker=ms[i], c="k", label="center"+str(i+1))
        plt.legend(loc="upper right")
        plt.title("FCM-Q"+str(q))


    print("\n**********实验" + str(q) + "**********")
    print("K-means的聚类中心为：\n", km.cluster_centers_, "\n迭代次数为：", km.n_iter)
    print("\nFCM的聚类中心为：\n", fcm.cluster_centers_, "\n迭代次数为：", fcm.n_iter)
plt.show()



# ii = np.array([[0,0,0,0], [1,1,1,1], [2,2,2,2]])
# data_iris_n = data_iris- data_iris.mean(axis=0, keepdims=True)
#
# from sklearn.cluster import KMeans as sk_km
# sk_km = sk_km(n_clusters=3)
# sk_km.fit(data_iris)
# print(sk_km.cluster_centers_, '\n', sk_km.labels_, '\n')
#
# # km = KMeans(n_clusters=3)
# # km.fit(data_iris)
# # print(km.cluster_centers_, '\n')
#
# fcm = FCM(n_clusters=3, b=1.1)#, init=sk_km.cluster_centers_)
# fcm.fit(data_iris)
# print(fcm.cluster_centers_, '\n', fcm.labels_, '\n')

