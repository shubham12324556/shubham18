# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:07:29 2023

@author: Dell
"""
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\50_Startups.csv')
df
Y = df[['Profit']]
X = df[['R&D Spend','Administration','Marketing Spend']]
type(X)
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y)
plt.xlabel('R&D Spend')
plt.show()
plt.scatter(X.iloc[:,1],Y)
plt.xlabel('Administration')
plt.show()
plt.scatter(X.iloc[:,2],Y)
plt.xlabel('Marketing Spend')
plt.show
df.corr()
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,2],X.iloc[:,1])
plt.xlabel('R&D Spend')
plt.ylabel('Administration')
plt.show()
plt.scatter(X.iloc[:,0],X.iloc[:,2])
plt.xlabel('Marketing Spend')
#plt.ylabel('State')
plt.show()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['State']=LE.fit_transform(df['State']) 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_



==================================================================================

import pandas as pd
import numpy as nd
df = pd.read_csv('C:\\Users\Dell\Downloads\\ToyotaCorolla.csv',encoding='latin1')
df
Y = df['Price']
X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]






