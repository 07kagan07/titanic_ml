# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
df_train.head()

df_train.shape

df_train.describe()

df_train.info()

sns.countplot(x = 'Survived', data = df_train, hue = 'Pclass')

sns.distplot(df_train['Age'], kde= False)

df_train.isna().sum()

"""## So we have missing values in age column. We should fill in these values with the average age of the entire dataset. Normally we should not do that but this is the only chance right now."""

print(df_train[df_train['Pclass'] == 1]['Age'].mean())
print(df_train[df_train['Pclass'] == 2]['Age'].mean())
print(df_train[df_train['Pclass'] == 3]['Age'].mean())

def fill_the_na_values(cols):
  age= cols[0]
  pclass = cols[1]

  if pd.isna(age):
    if pclass == 1:
      return round(df_train[df_train['Pclass'] == 1]['Age'].mean())
    if pclass == 2:
      return round(df_train[df_train['Pclass'] == 2]['Age'].mean())
    if pclass == 3:
      return round(df_train[df_train['Pclass'] == 3]['Age'].mean())
  else: 
    return age

df_train['Age'] = df_train[['Age','Pclass']].apply(fill_the_na_values, axis=1)

sns.boxplot(x ='Pclass', y= 'Age', data = df_train)

"""So older people have more wealth and thus they are in first class. """

df_train.isna().sum()

sns.heatmap(df_train.isna())

df_train.drop(['Cabin'],axis=1, inplace=True)

df_train.head()

df_train.dropna(inplace= True)

df_train.isna().sum()

df_train.drop(['PassengerId','Name','Ticket'], axis= 1, inplace = True)

df_train.head()

df_train.Sex.unique()

df_train.Embarked.unique()

df_train['Sex'] = df_train['Sex'].replace({'male': 0, 'female': 1})

Embarked = pd.get_dummies(df_train.Embarked, prefix= 'Embarked')
Embarked

df_train = df_train.join(Embarked)  
df_train.drop(['Embarked'],axis=1,inplace = True)

df_train.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x = df_train.drop('Survived', axis=1)
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train,y_train)

predictions = svm.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0,5,1,10,50,100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001,0.00001,0.000001]}

grid = GridSearchCV(SVC(),param_grid,refit = True,verbose= 2)
grid.fit(x_train,y_train)
grid_predictions = grid.predict(x_test)

print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test, grid_predictions))

##In this output, 0 and 1 represent class labels. In this example, 0 represents the negative class while 1 represents the positive class. 
##The values in the top left corner of the confusion matrix (47) are called true negatives, while the values in the top right corner (8) are called false positives. 
##The values in the bottom left corner (5) are called false negatives, while the values in the bottom right corner (29) are called true positives.

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)
lr_predictions = lr.predict(x_test)

print(classification_report(y_test,lr_predictions))
print(confusion_matrix(y_test, lr_predictions))

## In the Logistic Regression model, precision values are lower for class 1 but recall values are higher. F1-score values are also lower for both classes.

## Both models have a similar accuracy value, but it can be said that the classification performance is more balanced in the SVC model.

from sklearn.neighbors import KNeighborsClassifier

error_list = []

for i in range(1,40):
  knn = KNeighborsClassifier(n_neighbors = i)
  knn.fit(x_train,y_train)
  knn_predictions = knn.predict(x_test)
  error_list.append(np.mean(knn_predictions != y_test))

plt.plot(range(1,40),error_list)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
knn_predictions = knn.predict(x_test)

print(classification_report(y_test,knn_predictions))
print(confusion_matrix(y_test, knn_predictions))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt = DecisionTreeClassifier()
rfc = RandomForestClassifier(n_estimators=200)

dt.fit(x_train, y_train)
rfc.fit(x_train,y_train)

dt_predictions = dt.predict(x_test)
rfc_predictions = rfc.predict(x_test)

print(classification_report(y_test,dt_predictions))
print(confusion_matrix(y_test, dt_predictions))

"""We measured the classification performance of the decision tree model using the classification_report and confusion_matrix functions."""

print(classification_report(y_test,rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))

"""We measured the classification performance of the random forest model using the classification_report and confusion_matrix functions."""

#Done.