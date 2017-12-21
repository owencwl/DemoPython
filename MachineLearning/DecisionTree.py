from  sklearn.feature_extraction import DictVectorizer

import csv

from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

import pydot

# 打开csv文件 ； r'AllElectronics.csv'-> 如果字符串前面加 ‘r’ 表示 此字符串不发生转义，有转义字符时不发生转义，原封不动。
alldata=open('AllElectronics.csv','r')
reader=csv.reader(alldata)
headers =next(reader)

featureList=[]
labelList=[]


# 把原始数据 转为字典形式 放入数组中
for row in reader:
    labelList.append(row[len(row)-1])
    rowdict={}

    for i in range(1,len(row)-1):
        rowdict[headers[i]]=row[i]
    featureList.append(rowdict)

# print(labelList)
# print(featureList)


# 把特征值转换 0 1的这种形式
vector=DictVectorizer()
x=vector.fit_transform(featureList).toarray()

print(str(x))
print(vector.get_feature_names())

# 分类标签转换
lb=preprocessing.LabelBinarizer()
y=lb.fit_transform(labelList)

# print(y)

# 使用决策分类树  criterion默认为gini -> 使用的是CART决策树； 若criterion=entropy 则为ID3决策树   c4.5使用的是信息增益率
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(x,y)


# print(clf)
#绘制决策树 图 其中需要安装graphviz，配置环境变量
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("iris.pdf")
graph[0].write_png("iris.png")





# 修改值，做预测

oneRowX = x[0, :]
print("oneRowX: " + str(oneRowX))
newRowX = oneRowX
newRowX[1] = 1
newRowX[2] = 0
newRowX[3] = 1
newRowX[4] = 0
newRowX[8] = 0
newRowX[9] = 1
print("newRowX: " + str(newRowX))

predictedY = clf.predict([newRowX])
print("predictedY: " + str(predictedY))