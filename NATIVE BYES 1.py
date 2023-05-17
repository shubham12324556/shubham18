# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:14:47 2023

@author: Dell
"""
QUE  1


import pandas as pd

df=pd.read_csv('C:\\Users\Dell\Downloads\\SalaryData_Train.csv')
df

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in range(0,14):
    df.iloc[:,i]=LE.fit_transform(df.iloc[:,i])

Y=df['Salary']
X=df.iloc[:,:13]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,Y_train)
Y_pred_train=MNB.predict(X_train)
Y_pred_test=MNB.predict(X_test)

from sklearn.metrics import accuracy_score
print('Training Accuracy is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy is',accuracy_score(Y_test,Y_pred_test).round(2))


# 2)




df=pd.read_csv('C:\\Users\Dell\Downloads\\SalaryData_Test.csv')
df

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in range(0,14):
    df.iloc[:,i]=LE.fit_transform(df.iloc[:,i])

Y=df['Salary']
X=df.iloc[:,:13]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,Y_train)
Y_pred_train=MNB.predict(X_train)
Y_pred_test=MNB.predict(X_test)

from sklearn.metrics import accuracy_score
print('Training Accuracy is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy is',accuracy_score(Y_test,Y_pred_test).round(2))

