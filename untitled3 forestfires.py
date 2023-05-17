# -*- coding: utf-8 -*-
"""
Created on Wed May 17 07:49:40 2023

@author: Dell
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

forestfires = pd.read_csv("C:\\Users\Dell\Downloads\\forestfires (2).csv")

data = forestfires.describe()

##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)

##Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) # Accuracy = 100%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) #Accuacy = 100%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) #Accuracy = 74.6%

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 73%
