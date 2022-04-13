from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def kmeans(X, k):
    kmean = KMeans(n_clusters=k)
    return kmean.fit_predict(X)


def dbscan(X, eps):
    return DBSCAN(eps=eps, min_samples=5).fit_predict(X)


def euclid(a, b):
    s = 0
    for i in range(len(a)):
        s += pow(a[i] - b[i], 2)
    return sqrt(s)


def elbow(points, labels):
    print(labels)
    clusters = np.array([points[labels == i] for i in set(labels)], dtype=object)
    centroids = np.array([cluster.mean(axis=0) for cluster in clusters])
    res = 0
    for i in range(len(clusters)):
        s = 0
        for point in clusters[i]:
            s += euclid(point, centroids[i]) ** 2
        res += s
    return res / len(clusters)


def a(point, cluster):
    s = 0
    for p in cluster:
        s += euclid(p, point)
    return s / len(cluster)


def b(point, clusterIndex, clusters):
    l = []
    for i in range(len(clusters)):
        if i != clusterIndex:
            s = 0
            for p in clusters[i]:
                s += euclid(p, point)
            l.append(s / len(clusters[i]))
    if len(l) == 0:
        return 0
    return min(l)


def silhouette(points, labels):
    clusters = np.array([points[labels == i] for i in set(labels)], dtype=object)
    sum = 0
    for i in range(len(clusters)):
        for point in clusters[i]:
            b_res = b(point, i, clusters)
            a_res = a(point, clusters[i])
            sum += (b_res - a_res) / max(a_res, b_res)

    return sum / len(points)


def DBI(points, labels):
    clusters = np.array([points[labels == i] for i in set(labels)], dtype=object)
    print(len(clusters))
    centroids = np.array([cluster.mean(axis=0) for cluster in clusters])

    max_sum = 0

    for i in range(len(clusters)):
        arr = []
        for j in range(len(clusters)):
            if i != j:
                arr.append(
                    (a(centroids[i], clusters[i]) + a(centroids[j], clusters[j])) /
                    (euclid(centroids[i], centroids[j]))
                )
        if len(arr) > 0:
            max_sum += max(arr)
    return max_sum / len(clusters)

def neighbours_distances(points, n_neighbours):
    res = list()
    for i in range(len(points)):
        dists = list()
        for j in range(len(points)):
            if i != j:
                dists.append(euclid(points[i], points[j]))
        dists.sort()
        dists = dists[:n_neighbours]
        res.append(np.mean(dists))
    res.sort()
    return res


data = np.genfromtxt('test.csv', delimiter=',')
data = data[1:]
#
#
DBI_res = list()
elbow_res = list()
silhouette_res = list()
max_clusters = 15
x = range(2, max_clusters + 1)
for k in x:
    res = kmeans(data, k)
    DBI_res.append(DBI(data, res))
    elbow_res.append(elbow(data, res))
    silhouette_res.append(silhouette(data, res))

    print(DBI_res)
    print(silhouette_res)
    print("\b\b\b\b\b\b%2.2f" % (100 * k / max_clusters) + '%')
print()


plt.figure(1)
plt.title("Davies–Bouldin Index")
plt.plot(x, DBI_res)
plt.xticks(x)
plt.figure(2)
plt.title("Elbow Method")
plt.plot(x, elbow_res)
plt.xticks(x)
plt.figure(3)
plt.title("Silhouette")
plt.plot(x, silhouette_res)
plt.xticks(x)
# plt.show()


data = MinMaxScaler().fit_transform(data)
print(data)

DBI_res.clear()
elbow_res.clear()
silhouette_res.clear()

n = 5
dists = neighbours_distances(data, n)
plt.figure(5)
plt.title("Epsilon")
plt.plot(dists)
plt.show()

eps_axis = [0.05 * i for i in range(24, 30)]
for eps in eps_axis:
    print(eps)
    res = dbscan(data, eps)
    DBI_res.append(DBI(data, res))
    elbow_res.append(elbow(data, res))
    silhouette_res.append(silhouette(data, res))
print()


plt.figure(6)
plt.title("Davies–Bouldin Index")
plt.plot(eps_axis, DBI_res)
plt.xticks(eps_axis)
plt.figure(7)
plt.title("Elbow Method")
plt.plot(eps_axis, elbow_res)
plt.xticks(eps_axis)
plt.figure(8)
plt.title("Silhouette")
plt.plot(eps_axis, silhouette_res)
plt.xticks(eps_axis)
plt.show()

# n = 5
# dists = neighbours_distances(data, n)
# plt.figure(4)
# plt.title("Epsilon")
# plt.plot(dists)
# plt.show()
