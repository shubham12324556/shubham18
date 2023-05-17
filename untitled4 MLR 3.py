# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:13:21 2023

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np





startup = pd.read_csv('C:\\Users\Dell\Downloads\\50_Startups (1).csv')
startup





startup.head()





startup.shape





startup.info()





startup.isna().sum()





startup = startup.rename({'R&D Spend':'RDS', 'Administration':'ADM', 'Marketing Spend':'MS'},axis=1)
startup





startup.corr()



sns.set_style(style='darkgrid')
sns.pairplot(startup)





model = smf.ols('Profit~RDS+ADM+MS', data = startup).fit()





model.params





model.tvalues





model.pvalues





(model.rsquared,model.rsquared_adj)





slr_a=smf.ols('Profit~ADM', data = startup).fit()
slr_a.tvalues, slr_a.pvalues
slr_a.summary()





slr_m = smf.ols('Profit~MS', data = startup).fit()
slr_m.tvalues, slr_m.pvalues
slr_m.summary()





slr_am = smf.ols('Profit~ADM+MS', data = startup).fit()
slr_am.tvalues, slr_am.pvalues
slr_am.summary()





rsq_r = smf.ols('RDS~ADM+MS',data = startup).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a = smf.ols('ADM~RDS+MS', data = startup).fit().rsquared
vif_a = 1/(1-rsq_a)

rsq_m = smf.ols('MS~RDS+ADM', data = startup).fit().rsquared
vif_m = 1/(1-rsq_m)

df1={'Variables':['RDS','ADM','MS'],'Vif':[vif_r,vif_a,vif_m]}
vif_df = pd.DataFrame(df1)
vif_df





import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line='q')
plt.title('Normal Q-Q plot of residuals')
plt.show()





list(np.where(model.resid<-20000))





def standard_values(vals):
    return (vals-vals.mean())/vals.std()





plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()





fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RDS", fig=fig)
plt.show()





fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, 'ADM', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'MS', fig = fig)
plt.show()

model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance





fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(startup)),np.round(c,3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()





(np.argmax(c), np.max(c))





from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()





startup.shape





k = startup.shape[1]
n = startup.shape[0]
leverage_cutoff = 3*((k + 1 )/n)
leverage_cutoff





startup[startup.index.isin([49])]





startup





startup1 = startup.drop(startup.index[[49]], axis = 0).reset_index(drop=True)
startup1





final_data = smf.ols('Profit~ RDS+ADM+MS', data = startup1).fit()
final_data.summary()





(final_data.rsquared, final_data.rsquared_adj)





new_data = pd.DataFrame({'RDS': 15860,'ADM':58236,'MS':852965}, index= [0])
new_data





final_data.predict(new_data)





y_pred = final_data.predict(startup1)
y_pred





table=pd.DataFrame({'Prep_Models': ['Model','Final_Model'],'Rsquared':[model.rsquared,final_data.rsquared]})
table 
========================================================================================================================
QUESTION 2 


df = pd.read_csv('C:\\Users\Dell\Downloads\\ToyotaCorolla (1).csv', encoding = 'ISO-8859-1')
df





df = pd.concat([df.iloc[:,2:4],df.iloc[:,6],df.iloc[:,8],df.iloc[:,12:14],df.iloc[:,15],df.iloc[:,16:18]], axis = 1)
df





df.info()





df.shape





df[df.duplicated()]





df = df.drop_duplicates().reset_index(drop=True)
df





df.describe()





df.isna().sum()





df.corr()





df





df.isna().sum()





sns.set_style(style='darkgrid')
sns.pairplot(df)





model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data = df).fit()





model.params





model.tvalues





model.pvalues





(model.rsquared,model.rsquared_adj)





slr_c = smf.ols('Price~cc', data = df).fit()
print(slr_c.tvalues, '\n', slr_c.pvalues)
slr_c.summary()





slr_d = smf.ols('Price~Doors',data =df).fit()
print(slr_d.tvalues,'\n',slr_d.pvalues)
slr_d.summary()





slr_cd = smf.ols('Price~cc+Doors', data = df).fit()
print(slr_cd.tvalues,'\n', slr_cd.pvalues)
slr_cd.summary()





rsq_age = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_age = 1/(1-rsq_age)

rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data =df).fit().rsquared
vif_km = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_hp = 1/(1-rsq_HP)

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_cc = 1/(1-rsq_cc)

rsq_d = smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_d = 1/(1-rsq_d)

rsq_g = smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_g = 1/(1-rsq_g)

rsq_qt = smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight', data = df).fit().rsquared
vif_qt = 1/(1-rsq_qt)

rsq_w = smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax', data = df).fit().rsquared
vif_w = 1/(1-rsq_w)

dir1 = pd.DataFrame({'Variable':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],
       'VIF':[vif_age, vif_km, vif_hp, vif_cc, vif_d, vif_g, vif_qt, vif_w]})
dir1





import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line='q')
plt.title("Normal Q_Q plot of Residuals")
plt.show()





list(np.where(model.resid>6000))





list(np.where(model.resid<-6000))





def standard_values(vals):
    return (vals-vals.mean())/vals.std()





plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual plot')
plt.xlabel('Standard Fitted Values')
plt.ylabel('Standard Residual Values')
plt.show()





fig = plt.figure(figsize = (15,9))
fig = sm.graphics.plot_regress_exog(model,'Age_08_04', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'KM', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'HP', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'cc', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Doors', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Gears', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Quarterly_Tax', fig = fig)
plt.show()





fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Weight', fig = fig)
plt.show()





model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance





fig = plt.subplots(figsize = (20,7))
plt.stem(np.arange(len(df)), np.round(c,3))
plt.xlabel('Row index')
plt.ylabel("column index")
plt.show()





(np.argmax(c), np.max(c))





from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()





df.shape


# In[44]:


a = df.shape[1]
s = df.shape[0]
leverage_cutoff = 3*((a+1)/s)





leverage_cutoff



df[df.index.isin([80])]





df1 = df





df1





df2 = df1.drop(df1.index[[80]], axis = 0).reset_index(drop=True)
df2





final_df = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df2).fit()
final_df.summary()





new_data = pd.DataFrame({'Age_08_04':15,"KM": 58256,'HP':85,"cc": 1500,"Doors": 4,"Gears":7,"Quarterly_Tax":75,'Weight':1500}, index=[0])
new_data





final_df.predict(new_data)





y_pred = final_df.predict(df2)
y_pred





table = pd.DataFrame({'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_df.rsquared]})
table


