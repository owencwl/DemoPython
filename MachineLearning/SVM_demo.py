

from sklearn import svm


x=[[2,0],[1,1],[2,3]]

y=[0,0,1]

# 建立模型 一般都是fit()函数
clf=svm.SVC(kernel='linear')
clf.fit(x,y)

print(clf)

# 支持向量的点
print(clf.support_vectors_)

# 支持向量点的索引   在原始数据中的位置
print(clf.support_)

# 找到多少个支持向量
print(clf.n_support_)


# 预测值
print(clf.predict([[0.5,1]]))
