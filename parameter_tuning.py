from tools import preprocess
import numpy as np

names,features,labels=preprocess("cancer.csv")
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(features)
X=scaler.transform(features)
from sklearn.model_selection import train_test_split,GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.4,random_state=42)

from sklearn.decomposition import PCA
trainacc=[]
testacc=[]
precision=[]
recall=[]
n=21
for i in range(1,n):
  pca=PCA(n_components=i)
  pca.fit(x_train)

  x_train_trans=pca.transform(x_train)
  x_test_trans=pca.transform(x_test)
  param={"kernel":["rbf","poly","linear"],"C":[1,10,100],"gamma":[0.1,1,10]}
  from sklearn.svm import SVC
  mod=SVC()
  clf=GridSearchCV(mod,param)
  clf.fit(x_train_trans,y_train.ravel())
  pred=clf.predict(x_test_trans)
  tp=len([1 for i,x in enumerate(pred) if x==1 and y_test[i]==1])
  fp=len([1 for i,x in enumerate(pred) if x==1 and y_test[i]==0])
  fn=len([1 for i,x in enumerate(pred) if x==0 and y_test[i]==1])
  precision.append(tp/(tp+fp))
  recall.append(tp/(tp+fn))
  trainacc.append(clf.score(x_train_trans,y_train))
  testacc.append(clf.score(x_test_trans,y_test))
  print("for",i,"principal component:")
  print(clf.best_estimator_)

plt.ylim(0.8,0.99)
plt.plot(list(range(1,n)),trainacc,color="b")
plt.plot(list(range(1,n)),testacc,color="r")
plt.savefig("accuracy.jpg")
plt.show()
componentsaccuracy=list(zip(list(range(1,n)),trainacc,testacc))
for x in componentsaccuracy:
  print(x)
#print(zip(list(range(1,n)),testacc))

plt.figure()
plt.plot(list(range(1,n)),precision,color="b")
plt.plot(list(range(1,n)),recall,color="r")
plt.savefig("precisionandrecall.jpg")
plt.show()
precandrec=list(zip(list(range(1,n)),precision,recall))
for x in precandrec:
  print(x)
