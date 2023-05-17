# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:07:29 2023

@author: Dell
"""
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\50_Startups.csv')
df
Y = df[['Profit']]
X = df[['R&D Spend','Administration,''Marketing Spend']]
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
Y_pred = LR.predict(X)

# metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,Y_pred)
print("Mean squarred error: ",mse.round(2))

import numpy as np
rmse = np.sqrt(mse)
print("Root Mean squarred error: ",rmse.round(2))
print("R square: ", r2_score(Y,Y_pred).round(2))

# Statsmodel for multicollinearity
# pip install statsmodels
import statsmodels.formula.api as smf
model = smf.ols('',data=df).fit()
model.summary()

import statsmodels.formula.api as smf 
model = smf.ols('Profit~R&D Spend,Administration,Marketing Spend',data=df).fit()
model.summary()

#Coefficients
model.params

#R squared values
model.rsquared.round(3)


==================================================================================

import pandas as pd
import numpy as nd
df = pd.read_csv('C:\\Users\Dell\Downloads\\ToyotaCorolla.csv',encoding='latin1')
df
Y = df['Price']
X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
type (X)

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y)
plt.xlabel('Age_08_04')
plt.show()
plt.scatter(X.iloc[:,1],Y)
plt.xlabel('KM')
plt.show()
plt.scatter(X.iloc[:,2],Y)
plt.xlabel('HP')
plt.show
df.corr()
mport matplotlib.pyplot as plt
plt.scatter(X.iloc[:,3],Y)
plt.xlabel('cc')
plt.show()
plt.scatter(X.iloc[:,4],Y)
plt.xlabel('Doors')
plt.show()
plt.scatter(X.iloc[:,5],Y)
plt.xlabel('Gears')
plt.show
df.corr()
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,2],X.iloc[:,1])
plt.xlabel('Age_08_04')
plt.ylabel('KM')
plt.show()
plt.scatter(X.iloc[:,0],X.iloc[:,2])
plt.xlabel('HP')
#plt.ylabel('CC')
plt.show()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['Price']=LE.fit_transform(df['Price']) 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)

# metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,Y_pred)
print("Mean squarred error: ",mse.round(2))

import numpy as np
rmse = np.sqrt(mse)
print("Root Mean squarred error: ",rmse.round(2))
print("R square: ", r2_score(Y,Y_pred).round(2))

# Statsmodel for multicollinearity
# pip install statsmodels
import statsmodels.formula.api as smf
model = smf.ols('Price~Age_08_04+KM+HP+CC',data=df).fit()
model.summary()

import statsmodels.formula.api as smf
model = smf.ols('MPG ~ HP+VOL',data=df).fit()
model.summary()

#Coefficients
model.params

#R squared values
model.rsquared.round(3)




