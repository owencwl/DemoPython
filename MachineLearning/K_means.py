import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
# maxIt最大的迭代次数
def kmeans(X, k, maxIt):

    # 得到 行、列 数
    numPoints, numDim = X.shape
    # 矩阵多加一列，并且全部赋值为0
    dataSet = np.zeros((numPoints, numDim + 1))
    # 把原始数据集赋值给新的数组dataSet,其中最后一列为0
    dataSet[:, :-1] = X

    # Initialize centroids randomly
    # 从矩阵每行中随机选取K行  也就是说 选k个中心点
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    # centroids = dataSet[0:2, :]
    # Randomly assign labels to initial centorid
    # 对k行的最后一列 赋值 k种类别
    centroids[:, -1] = range(1, k + 1)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    # shouldStop（）计算迭代停止 条件
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iteration: \n", iterations)
        print("dataSet: \n", dataSet)
        print("centroids: \n", centroids)

        # Save old centroids for convergence test. Book keeping.
        # numpy复制数组
        oldCentroids = np.copy(centroids)
        iterations += 1

        # Assign labels to each datapoint based on centroids
        # 对每行数据 更新类别数据 所属的类别需要不断的更新
        updateLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        # 重新获取中心点 计算平均值
        centroids = getCentroids(dataSet, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return dataSet


# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    # 注意数组的比较 使用numpy的array_equal函数
    return np.array_equal(oldCentroids, centroids)


# Function: Get Labels
# -------------
# Update a label for each piece of data in the dataset.
def updateLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)


def getLabelFromClosestCentroid(dataSetRow, centroids):
    label = centroids[0, -1]
    #norm 表示范数 默认为L2  这里算出了两个向量之间的距离
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            # 更新距离相近的点 的类别
            label = centroids[i, -1]
    print("minDist:", minDist)
    return label


# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getCentroids(dataSet, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        # 对两个点之间计算平均值，更新中心点的位置
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i

    return result


# 准备数据集
x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
x5=np.array([6,3])
x6=np.array([1,1.5])

# 合并为一个矩阵
# testX = np.vstack((x1, x2, x3, x4,x5,x6))
# testX=np.random.randint(1,100,size=[100,5])
testX, y = make_blobs(n_samples=1500, random_state=150)

#自己实现kmeans testx:数据集 3：K值 15：最大迭代次数
result = kmeans(testX, 3, 15)


# 绘制散点图
plt.figure(figsize=(10, 10))
# plt.subplot(221)
plt.scatter(result[:,0],result[:,1],result[:,2],c=result[:,-1])
plt.show()

print("final result:\n",result)
