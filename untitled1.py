# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:38:58 2023

@author: Dell
"""
Question 1

import numpy as np
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\Cutlets.csv') 
df.shape
df.head()
df.dtypes

xbar = df['Unit A'].mean()
mu = 8
s =df['Unit A'].std()
rootn = np.sqrt(len(df['Unit A']))
num = (xbar - mu)
den =s/rootn
zcalc = num/den
zcalc

# For any one sided alpha = 5%
import scipy.stats as stats
stats.norm.ppf(.95).round(2)

#### ONE SAMPLE Z TEST/ MEAN TEST

from statsmodels.stats import weightstats as onesample
Zcal,pval = onesample.ztest(df['Unit A'],value=8,alternative= 'two-sided')
print(Zcal)
help(onesample.ztest)
if pval<.05:
    print('H0 is rejected and H1 is accepted')
else:
    print("H1 is rejected and H0 is accepted")
=================================================================================    
QUESTION 2

import numpy as np
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\LabTAT.csv')
df
df['Laboratory 1'].mean()
df['Laboratory 2'].mean()
from scipy import stats
zcal,pval = stats.ttest_ind( df['Laboratory 1'], df['Laboratory 2'])
print('P-value is',pval.round(4))
if pval<.05:
    print('reject null hypothesis, Accept Alternative hypothesis')
else:
    print('accept null hypothesis, reject Alternative hypothesis')
print('Zcalculated value is', zcal.round(4))


###################################################################################

QUESTION 3

import pandas as pd
df =pd.read_csv('C:\\Users\Dell\Downloads\\BuyerRatio (1).csv')
df
df['East'].mean()
df['West'].mean()
from scipy import stats
zcal , pval = stats.ttest_ind(df['East'],df['West'])
print('P-value is ',pval.round(4))
if pval<.05:
    print('reject null hypothesis, Accept Alternative hypothesis')
else:
    print('accept null hyothesis, Reject Alternative hypothesis ')
print('Zcalculated value is ', zcal.round(4))


#####################################################################################

QUESTION 4
from scipy import stats
import numpy as np
import pandas as pd
df = pd.read_csv('C:\\Users\Dell\Downloads\\Costomer+OrderForm.csv')
df
df['India'].mean
#alpha = 10%--> 90%
df_ci = norm.interval(.90,loc=df['India'].mean, scale= df['India'].std())




