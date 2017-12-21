import  numpy as np
# 画图常用
import pylab as pl

from sklearn import svm

#
np.random.seed(0)

# 随机产生 点  符合正态分布
x=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]

y=[0]*20+[1]*20


clf=svm.SVC(kernel='linear')
clf.fit(x,y)

w=clf.coef_[0]
a=-w[0]/w[1]

xx=np.linspace(-5,5)

yy=a*xx-(clf.intercept_[0]/w[0])

b=clf.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])

b=clf.support_vectors_[-1]

yy_up=a*xx+(b[1]-a*b[0])



pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors='none')
pl.scatter(x[:,0],x[:,1],c=y,cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()