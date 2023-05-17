# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:06:43 2023

@author: Dell
"""

# 1) CocaCola Prices




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

import itertools
import statsmodels.api as sm

coke=pd.read_excel("C:\\Users\Dell\Downloads\\CocaCola_Sales_Rawdata (1).xlsx",index_col=0,parse_dates=True) 
coke

coke.info()

coke = pd.read_excel("C:\\Users\Dell\Downloads\\CocaCola_Sales_Rawdata (1).xlsx", index_col = 0,header = 0,parse_dates = True)
coke

coke.index

plt.figure(figsize = (15,7))
plt.plot(coke)

coke.plot(kind='kde')

coke.hist()

plt.figure(figsize = (17,7))
pd.plotting.lag_plot(coke)

from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (50,15))
plot_acf(coke, lags=6)
plt.show()

coke = pd.read_excel("C:\\Users\Dell\Downloads\\CocaCola_Sales_Rawdata (1).xlsx",index_col = 0,header = 0,parse_dates = True, squeeze=True)
coke

type(coke)

coke = pd.read_excel("CocaCola_Sales_Rawdata.xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
coke

coke.shape

coke = pd.read_excel("C:\\Users\Dell\Downloads\\CocaCola_Sales_Rawdata (1).xlsx")

quarter =['Q1','Q2','Q3','Q4']





p = coke["Quarter"][0]
p[0:2]
coke['quarter']= 0

for i in range(42):
    p = coke["Quarter"][i]
    coke['quarter'][i]= p[0:2]

coke

quarter_dummies = pd.DataFrame(pd.get_dummies(coke['quarter']))
quarter_dummies

coke=pd.concat([coke,quarter_dummies],axis=1)
coke

coke['t']=np.arange(1,43)
coke['t_square']=np.square(coke.t)
coke['log_Sales']=np.log(coke.Sales)
coke

coke

coke['Sales'].plot()

plt.figure(figsize=(12,4))
sns.lineplot(x="quarter",y="Sales",data=coke)

coke

Train = coke.head(25)
Test = coke.tail(7)

Train

Test





#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear





#Exponential
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp





#Quadratic 
Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


#Additive seasonality 
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1', 'Q2', 'Q3', 'Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=coke).fit()

pred_new  = pd.Series(model_full.predict(coke))
pred_new

coke["forecasted_Sales"] = pd.Series(np.exp(pred_new))

plt.figure(figsize=(18,10))
plt.plot(coke[['Sales','forecasted_Sales']].reset_index(drop=True))





Train = coke.head(35)
Test = coke.tail(7)





plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
coke["Sales"].rolling(4).mean().plot(label=str(5))
plt.legend(loc='best')





plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
for i in range(2,18,6):
    coke["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')





decompose_ts_add = seasonal_decompose(coke['Sales'], period = 12)
decompose_ts_add.plot()
plt.show()





import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(coke.Sales,lags=12)
tsa_plots.plot_pacf(coke.Sales,lags=12)
plt.show()





def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)





ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) 





hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) 





hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)





hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)





hwe_model_mul_add = ExponentialSmoothing(coke["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 





hwe_model_mul_add.forecast(7)








# 2) Airline Passengers Data Set




airlines = pd.read_excel('C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx',index_col=0,parse_dates=['Month'])
airlines


airlines.info()

airlines.index

plt.figure(figsize = (15,7))
plt.plot(airlines)

airlines = pd.read_excel("C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx",index_col = 0,header = 0, parse_dates = True)
airlines

airlines.hist()

airlines.plot(kind='kde')

airlines = pd.read_excel("C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
airlines


type(airlines)





groups = airlines.groupby(pd.Grouper(freq='A'))
groups





years = pd.DataFrame()

for name, group in groups:
    years[name.year] = group.values

years





plt.figure(figsize = (15,7))
years.boxplot()





plt.figure(figsize = (15,9))
pd.plotting.lag_plot(airlines)





from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (32,20))
plot_acf(airlines, lags=95)
plt.show()





airlines = pd.read_excel("Airlines+Data.xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
airlines





airlines.shape





upsampled = airlines.resample('D').mean()
upsampled.head(20)





upsampled.shape



interpolated = upsampled.interpolate(method='linear')
interpolated.head(30)





airlines.plot()





resample = airlines.resample('Q')
quarterly_mean_sales = resample.mean()

quarterly_mean_sales.plot()

airlines=pd.read_excel("C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx",index_col=0,header=0, parse_dates=True)
airlines





# line plot
plt.subplot(211)
plt.plot(airlines)





# histogram
plt.subplot(212)
plt.hist(airlines)





dataframe = pd.DataFrame(np.log(airlines.values), columns = ['Passengers'])
dataframe





# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])





# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()





quarterly_mean_sales.head()





dataframe = pd.DataFrame(np.sqrt(airlines.values), columns = ['Passengers'])
dataframe





# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])





# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()





airlines=pd.read_excel("C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx")
airlines





airlines['Passengers'].plot()

airlines["month"] = airlines['Month'].dt.strftime("%b")
airlines["year"] = airlines['Month'].dt.strftime("%Y")

airlines

mp = pd.pivot_table(data = airlines,values = "Passengers",index = "year",columns = "month",aggfunc = "mean",fill_value=0)
mp

plt.figure(figsize=(12,8))
sns.heatmap(mp,annot=True,fmt="g",cmap = 'YlGnBu')

plt.figure(figsize=(15,10))

plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=airlines)

plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=airlines)

plt.figure(figsize=(17,8))
sns.lineplot(x="year",y="Passengers",data=airlines)





airlines

airlines.shape

airlines['t']=np.arange(1,97)
airlines['t_square']=np.square(airlines.t)
airlines['log_Passengers']=np.log(airlines.Passengers)
airlines2=pd.get_dummies(airlines['month'])

airlines

airlines2

airlines=pd.concat([airlines,airlines2],axis=1)
airlines

Train = airlines.head(84)
Test = airlines.tail(12)

Train

Test

#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


#Exponential
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#Quadratic 
Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality 
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea





#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

model_full = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airlines).fit()

pred_new  = pd.Series(model_full.predict(airlines))
pred_new

airlines["forecasted_Passengers"] = pd.Series(np.exp(pred_new))

plt.figure(figsize=(15,10))
plt.plot(airlines[['Passengers','forecasted_Passengers']].reset_index(drop=True))

airlines=pd.read_excel("C:\\Users\Dell\Downloads\\Airlines+Data (1).xlsx")
Train = airlines.head(84)
Test = airlines.tail(12)

Train

Test

plt.figure(figsize=(24,7))
airlines['Passengers'].plot(label="org")
airlines["Passengers"].rolling(15).mean().plot(label=str(5))
plt.legend(loc='best')

plt.figure(figsize=(24,7))
airlines['Passengers'].plot(label="org")
for i in range(2,24,6):
    airlines["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

decompose_ts_add = seasonal_decompose(airlines['Passengers'], period = 12)
decompose_ts_add.plot()
plt.show()


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(airlines.Passengers,lags=12)
tsa_plots.plot_pacf(airlines.Passengers,lags=12)
plt.show()

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)





ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)

hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) 

hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) 

hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)

hwe_model_mul_add = ExponentialSmoothing(airlines["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 


hwe_model_mul_add.forecast(12)

