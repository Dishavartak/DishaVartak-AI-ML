import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#print(df)
df=pd.read_csv("D:/Disha/Diploma/AI ML/archive 1/boston.csv")

'''
df['CRIM']=pd.cut(df['CRIM'],3,labels=['A','B','C'])
df['ZN']=pd.cut(df['ZN'],3,labels=['A','B','C'])
df['INDUS']=pd.cut(df['INDUS'],3,labels=['A','B','C'])
df['CHAS']=pd.cut(df['CHAS'],3,labels=['A','B','C'])
'''
# to check if any value is missing

print(df.isnull() .sum())
df['CRIM'] .fillna((df['CRIM'].mean()), inplace=True)
print(df.isnull() .sum())

X=df.drop('CRIM',axis=1)
X=X.drop('ZN',axis=1)
Y=df['ZN']
print(Y)
le=LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)

#checking outliers

sns.boxplot(y='CRIM', data = df)
plt.title("Box Plot")
plt.show()

print(df['CRIM'])
Q1 = df['CRIM'] .quantile(0.25)
Q3 = df['CRIM'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['CRIM'] < lower].values
out2=df[df['CRIM'] < lower].values

df['CRIM'].replace(out1,lower)
df['CRIM'].replace(out2,upper)
plt.show()

sns.boxplot(y='ZN', data = df)
plt.title("Box Plot")
plt.show()

print(df['ZN'])
Q1 = df['ZN'] .quantile(0.25)
Q3 = df['ZN'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['ZN'] < lower].values
out2=df[df['ZN'] < lower].values

df['ZN'].replace(out1,lower)
df['ZN'].replace(out2,upper)

sns.boxplot(y='INDUS', data = df)
plt.title("Box Plot")
plt.show()
print(df['INDUS'])
Q1 = df['INDUS'] .quantile(0.25)
Q3 = df['INDUS'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['INDUS'] < lower].values
out2=df[df['INDUS'] < lower].values

df['INDUS'].replace(out1,lower)
df['INDUS'].replace(out2,upper)

sns.boxplot(y='CHAS', data = df)
plt.title("Box Plot")
plt.show()

print(df['CHAS'])
Q1 = df['CHAS'] .quantile(0.25)
Q3 = df['CHAS'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['CHAS'] < lower].values
out2=df[df['CHAS'] < lower].values

df['CHAS'].replace(out1,lower)
df['CHAS'].replace(out2,upper)

sns.boxplot(y='NOX', data = df)
plt.title("Box Plot")
plt.show()
print(df['NOX'])
Q1 = df['NOX'] .quantile(0.25)
Q3 = df['NOX'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['NOX'] < lower].values
out2=df[df['NOX'] < lower].values

df['NOX'].replace(out1,lower)
df['NOX'].replace(out2,upper)

sns.boxplot(y='RM', data = df)
plt.title("Box Plot")
plt.show()
print(df['RM'])
Q1 = df['RM'] .quantile(0.25)
Q3 = df['RM'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['RM'] < lower].values
out2=df[df['RM'] < lower].values

df['RM'].replace(out1,lower)
df['RM'].replace(out2,upper)

sns.boxplot(y='AGE', data = df)
plt.title("Box Plot")
plt.show()
print(df['AGE'])
Q1 = df['AGE'] .quantile(0.25)
Q3 = df['AGE'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['AGE'] < lower].values
out2=df[df['AGE'] < lower].values

df['AGE'].replace(out1,lower)
df['AGE'].replace(out2,upper)

sns.boxplot(y='DIS', data = df)
plt.title("Box Plot")
plt.show()
print(df['DIS'])
Q1 = df['DIS'] .quantile(0.25)
Q3 = df['DIS'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['DIS'] < lower].values
out2=df[df['DIS'] < lower].values

df['DIS'].replace(out1,lower)
df['DIS'].replace(out2,upper)

sns.boxplot(y='RAD', data = df)
plt.title("Box Plot")
plt.show()
print(df['RAD'])
Q1 = df['RAD'] .quantile(0.25)
Q3 = df['RAD'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['RAD'] < lower].values
out2=df[df['RAD'] < lower].values

df['RAD'].replace(out1,lower)
df['RAD'].replace(out2,upper)

sns.boxplot(y='TAX', data = df)
plt.title("Box Plot")
plt.show()
print(df['TAX'])
Q1 = df['TAX'] .quantile(0.25)
Q3 = df['TAX'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['TAX'] < lower].values
out2=df[df['TAX'] < lower].values

df['TAX'].replace(out1,lower)
df['TAX'].replace(out2,upper)

sns.boxplot(y='PTRATIO', data = df)
plt.title("Box Plot")
plt.show()
print(df['PTRATIO'])
Q1 = df['PTRATIO'] .quantile(0.25)
Q3 = df['PTRATIO'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['PTRATIO'] < lower].values
out2=df[df['PTRATIO'] < lower].values

df['PTRATIO'].replace(out1,lower)
df['PTRATIO'].replace(out2,upper)

sns.boxplot(y='B', data = df)
plt.title("Box Plot")
plt.show()
print(df['B'])
Q1 = df['B'] .quantile(0.25)
Q3 = df['B'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['B'] < lower].values
out2=df[df['B'] < lower].values

df['B'].replace(out1,lower)
df['B'].replace(out2,upper)

sns.boxplot(y='LSTAT', data = df)
plt.title("Box Plot")
plt.show()
print(df['LSTAT'])
Q1 = df['LSTAT'] .quantile(0.25)
Q3 = df['LSTAT'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['LSTAT'] < lower].values
out2=df[df['LSTAT'] < lower].values

df['LSTAT'].replace(out1,lower)
df['LSTAT'].replace(out2,upper)

sns.boxplot(y='MEDV', data = df)
plt.title("Box Plot")
plt.show()
print(df['MEDV'])
Q1 = df['MEDV'] .quantile(0.25)
Q3 = df['MEDV'] .quantile(0.75)
IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)

out1=df[df['MEDV'] < lower].values
out2=df[df['MEDV'] < lower].values

df['MEDV'].replace(out1,lower)
df['MEDV'].replace(out2,upper)

reg=LinearRegression()
x=df.drop('MEDV',axis=1)
y=df['MEDV']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)
train=reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
score=mean_squared_error(Y_test,Y_pred)
print(score)