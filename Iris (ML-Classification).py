'''import pandas as pd
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
print(confusion_matrix(Y_test,y_pred)'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("D:/Disha/Diploma/AI ML/Iris.csv")

# Prepare the data
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(Y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(Y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, y_pred_rf))

# Gradient Boosting Machine
gbm = GradientBoostingClassifier(n_estimators=10)
gbm.fit(X_train, Y_train)
y_pred_gbm = gbm.predict(X_test)
print("GBM Accuracy:", accuracy_score(Y_test, y_pred_gbm))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(Y_test, y_pred_nb))

# Logistic Regression
logr = LogisticRegression(max_iter=200)
logr.fit(X_train, Y_train)
y_pred_logr = logr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_pred_logr))

# Classification Report and Confusion Matrix for Logistic Regression
print("Classification Report for Logistic Regression:\n", classification_report(Y_test, y_pred_logr))
print("Confusion Matrix for Logistic Regression:\n", confusion_matrix(Y_test, y_pred_logr))


