# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:42:49 2023

@author: Dell
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('C:\\Users\Dell\Downloads\\Company_Data.csv')
df

df.describe()

df['Sales']=pd.cut(x=df['Sales'],bins=[0,3,6,14],labels=['low','medium','high'],right=False)
df['CompPrice']=pd.cut(x=df['CompPrice'],bins=[77,100,135,176],labels=['low','medium','high'],right=False)
df['Income']=pd.cut(x=df['Income'],bins=[21,70,92,121],labels=['low','medium','high'],right=False)
df['Advertising']=pd.cut(x=df['Advertising'],bins=[0,2.5,13,30],labels=['low','medium','high'],right=False)
df['Population']=pd.cut(x=df['Population'],bins=[10,200,400,510],labels=['low','medium','high'],right=False)
df['Price']=pd.cut(x=df['Price'],bins=[24,110,137,195],labels=['low','medium','high'],right=False)
df['Age']=pd.cut(x=df['Age'],bins=[25,48,66,81],labels=['low','medium','high'],right=False)
df['Education']=pd.cut(x=df['Education'],bins=[10,13,16,19],labels=['low','medium','high'],right=False)

df

df.shape

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['ShelveLoc'])
plt.show()
sns.countplot(df['Urban'])
plt.show()
sns.countplot(df['US'])
plt.show()
sns.countplot(df['Sales'])
plt.show()

df.info()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in df.iloc[:,:]:
    df[i]=LE.fit_transform(df[i])

df

Y=df['Sales']
X=df.drop('Sales',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier()
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))

a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier(criterion='entropy',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))

import graphviz
from sklearn import tree

plt.show(tree.plot_tree(DTC))

fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))

np.mean(Y_pred_test==Y_test)

a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier(criterion='gini',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))

fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))

np.mean(Y_pred_test==Y_test)

RFC=RandomForestClassifier(n_estimators=500,max_samples=0.7,max_features=0.7,random_state=27)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=27)

RFC.fit(X_train,Y_train)

Y_pred_train=RFC.predict(X_train)
Y_pred_test=RFC.predict(X_test)





print('Random Forest Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Random Forest Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))

================================================================================================








df=pd.read_csv('C:\\Users\Dell\Downloads\\Fraud_check.csv')
df

df.describe()

df['Taxable.Income']=pd.cut(x=df['Taxable.Income'],bins=[0,30000,100000],labels=['Risky','Good'],right=False)
df

df['Taxable.Income'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['Undergrad'])
plt.show()
sns.countplot(df['Marital.Status'])
plt.show()
sns.countplot(df['Urban'])
plt.show()
sns.countplot(df['Taxable.Income'])
plt.show()

df.info()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in df.iloc[:,:]:
    df[i]=LE.fit_transform(df[i])

Y=df['Taxable.Income']
X=df.drop('Taxable.Income',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier()
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))

c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier(criterion='entropy',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))

import graphviz
from sklearn import tree

plt.show(tree.plot_tree(DTC))





fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))

np.mean(Y_pred_test==Y_test)

c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
    DTC=DecisionTreeClassifier(criterion='gini',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))

fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))

np.mean(Y_pred_test==Y_test)

RFC=RandomForestClassifier(n_estimators=500,max_samples=0.7,max_features=0.7,random_state=29)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=29)

RFC.fit(X_train,Y_train)

Y_pred_train=RFC.predict(X_train)
Y_pred_test=RFC.predict(X_test)

print('Random Forest Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Random Forest Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))

