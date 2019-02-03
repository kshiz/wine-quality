

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn import metrics
df =pd.read_csv('Iris.csv')
train, test =train_test_split(df,test_size=0.3)
train_x=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_x=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_y=test.Species
#Logistic Regression
model=LogisticRegression()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('Accuracy=',metrics.accuracy_score(prediction,test_y))
#Decision Tree
dt =DecisionTreeClassifier(max_depth=5)
dt.fit(train_x,train_y)
prediction=dt.predict(test_x)
print('Accuraacy=',metrics.accuracy_score(prediction,test_y))
#RandomForest classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
rf.fit(train_x, train_y)
rf_predict=rf.predict(test_x)
rf_acc_score = metrics.accuracy_score(test_y, rf_predict)
print(rf_acc_score)