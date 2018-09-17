#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:33:47 2018

@author: kshitij
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv('redwine.csv')
df.head()
fig = plt.figure(figsize=(6, 4))
x0=df['fixed acidity']
y0=df['quality']
plt.scatter(x0,y0,color='red')

#Create an empty list called Reviews
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews
x = df.iloc[:,:11]
y = df['Reviews']
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)
#decisiontree classiffication
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_acc_score = metrics.accuracy_score(y_test, dt_predict)
print(dt_acc_score*100)
#RandomForest classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)
rf_acc_score = metrics.accuracy_score(y_test, rf_predict)
print(rf_acc_score*100)