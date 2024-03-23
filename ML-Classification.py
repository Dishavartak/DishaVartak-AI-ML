import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
logr=LogisticRegression()
df=pd.read_csv("D:/Disha/Diploma/AI ML/Iris.csv")

x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']

knn=KNeighborsClassifier(n_neighbors=5)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0)
train=knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
print(accuracy_score(Y_test,y_pred))


dt=tree.DecisionTreeClassifier()
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.3)
train=dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
print(accuracy_score(Y_test,y_pred))


rf=RandomForestClassifier()
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.3)
train=rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print(accuracy_score(Y_test,y_pred))


gbm=GradientBoostingClassifier(n_estimators=10)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)
gbm.fit(X_train,Y_train)
y_pred=gbm.predict(X_test)
print(accuracy_score(Y_test,y_pred))

nb=GaussianNB
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.3)
nb.fit(X_train,Y_train)
Y_pred1=nb.predict(X_test)
print("Naive Bayes:",accuracy_score(Y_test,Y_pred1))
logr.fit(X_train,Y_train)
y_pred=logr.predict(X_test)
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))