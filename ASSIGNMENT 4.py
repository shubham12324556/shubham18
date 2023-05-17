# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:48:55 2023

@author: Dell
"""
QUESTION 1
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\Salary_Data (1).csv')
df

Y = df['YearsExperience']
X = df[['Salary']]
import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.show()
df.corr()
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
# B0
LR.intercept_
#B1
LR.coef_
# B0 + B1(Salary)
Y_pred = LR.predict(X)
Y_pred

import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.scatter(X, Y_pred,color= 'red')
plt.plot(X['Salary'],Y_pred,color='black')
plt.show()
# metrics

#######################################################################################


QUESTION 2

import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\delivery_time.csv')
df
Y = df['Delivery Time']
X = df[['Sorting Time']]
import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.show()
df.corr()

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

#B0
LR.intercept_
# B1
LR.coef_
#B0 + B1(Sorting Time)
Y_pred = LR.predict(X)
Y_pred
import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X['Sorting Time'],Y_pred, color= 'blue')
plt.show()
# metrics
