# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:03:18 2023

@author: Dell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

bank = pd.read_csv('C:\\Users\Dell\Downloads\\bank-full.csv.crdownload')
bank
bank.info()
data1=pd.get_dummies('bank,columns'=="[,'job','marital','education','default','balance','housing','loan','contact','day','month','duration','campagin','pdays','previous','y']")
data1
pd.set_option("display.max.columns",None)
data1
data1.info()
data1["default"]= np.where(data1["default"].str.contains("yes"),1,0)
data1['housing']= np.where(data1['housing'].str.contains("yes"),1,0)
data1['loan']= np.where(data1['loan'].str.contains("yes"),1,0)
data1['y']= np.where(data1['y'].str.contains("yes"),1,0)
data