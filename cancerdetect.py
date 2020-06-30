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


pca=PCA(n_components=11)
pca.fit(x_train)

x_train_trans=pca.transform(x_train)
x_test_trans=pca.transform(x_test)
from sklearn.svm import SVC
clf=SVC(kernel="linear",C=10,gamma=0.1)
clf.fit(x_train_trans,y_train.ravel())
pred=clf.predict(x_test_trans)
tp=len([1 for i,x in enumerate(pred) if x==1 and y_test[i]==1])
fp=len([1 for i,x in enumerate(pred) if x==1 and y_test[i]==0])
fn=len([1 for i,x in enumerate(pred) if x==0 and y_test[i]==1])
precision=(tp/(tp+fp))
recall=(tp/(tp+fn))
trainacc=(clf.score(x_train_trans,y_train))
testacc=(clf.score(x_test_trans,y_test))


print("training set accuracy:",trainacc)
print("test set accuracy:",testacc)


print("precision:",precision)
print("recall:",recall)
