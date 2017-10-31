from numpy import *
import sys
import operator

def createdataset():
    group=array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels=['A','A','B','B']
    return group,labels

def classify(input,dataset,label,k):
    datasize=dataset.shape[0]

    ####计算欧式距离
    diff = tile(input, (datasize, 1)) - dataset
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  ###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5


    ##对距离进行排序
    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes



dataSet,labels = createdataset()
input = array([0.1,3.6])
K = 3
output = classify(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)