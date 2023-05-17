# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:08:53 2023

@author: Dell
"""

pip install mlxtend

import pandas as pd
df = pd.read_csv("C:\\Users\Dell\Downloads\\SalaryData_Test(1).csv")
df.shape
df.head()

X = df.iloc[:,0:2]
y = df.iloc[:,3]

# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

# Training a classifier - kernel=linear
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred_test)

print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(2))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=y.values,
                      clf=clf, 
                      legend=3)

=======================================================================================================================




pip install mlxtend

import pandas as pd
df = pd.read_csv("C:\\Users\Dell\Downloads\\SalaryData_Train(1).csv")
df.shape
df.head()

X = df.iloc[:,0:2]
y = df.iloc[:,3]

# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

# Training a classifier - kernel=linear
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred_test)

print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(2))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=y.values,
                      clf=clf, 
                      legend=3)


===============================================================================================================

pip install mlxtend

import pandas as pd
df = pd.read_csv("C:\\Users\Dell\Downloads\\forestfires.csv")
df.shape
df.head()

X = df.iloc[:,0:2]
y = df.iloc[:,3]

# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

# Training a classifier - kernel=linear
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred_test)

print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(2))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=y.values,
                      clf=clf, 
                      legend=3)


======================================================================================================





