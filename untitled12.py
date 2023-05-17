# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:44:43 2023

@author: Dell
"""
QUESTION 1


import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\Dell\Downloads\\Zoo (1).csv')
df
Y = df['animal name']
from sklearn.preprocessing import LabelEncoder
LE =LabelEncoder()
Y = LE.fit_transform(Y)
X = df.iloc[:,2:]
list(X)
from sklearn.preprocessing import StandardScaler
SS =StandardScaler()
SS_X = SS.fit_transform(X)
SS_X
SS_X = pd.DataFrame(SS_X)
SS_X.columns = list(X)
SS_X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,train_size=.7)
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9,p=2)
KNN.fit(X_train, Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

from sklearn.metrics import accuracy_score
print('Training Accuracyscore:',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy score:',accuracy_score(Y_test,Y_pred_test).round(2))


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_test = logreg.predict(X_test)
Y_pred_train =logreg.predict(X_train)
from sklearn.metrics import accuracy_score
print('Training Accuracyscore:',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy score:',accuracy_score(Y_test,Y_pred_test).round(2))


=================================================================================================

