# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:36:24 2023

@author: Dell
"""

import pandas as pd

df=pd.read_csv("C:\\Users\Dell\Downloads\\bank-full.csv", sep=';')
df

df.dtypes

from sklearn.preprocessing import LabelEncoder,StandardScaler

LE=LabelEncoder()

df['y']=LE.fit_transform(df['y'])
df['default']=LE.fit_transform(df['default'])
df['housing']=LE.fit_transform(df['housing'])
df['loan']=LE.fit_transform(df['loan'])
df['job']=LE.fit_transform(df['job'])
df['marital']=LE.fit_transform(df['marital'])
df['education']=LE.fit_transform(df['education'])
df['job']=LE.fit_transform(df['job'])
df['contact']=LE.fit_transform(df['contact'])
df['month']=LE.fit_transform(df['month'])
df['poutcome']=LE.fit_transform(df['poutcome'])

df

df.dtypes

df.shape

Y=df['y']
X=df.iloc[:,1:16]

X

from sklearn.linear_model import LogisticRegression

Logreg=LogisticRegression(max_iter=30000)
Logreg.fit(X,Y)

Y_pred=Logreg.predict(X)

from sklearn.metrics import confusion_matrix,accuracy_score

CM=confusion_matrix(Y,Y_pred)
CM

AS=accuracy_score(Y,Y_pred)
print('Accuracy Score is',AS.round(3))

from sklearn.metrics import recall_score,precision_score,f1_score,roc_curve,roc_auc_score
RS=recall_score(Y,Y_pred)
print('Sensitivity Score is',RS.round(2))

true_negative=CM[0][0]
false_positive=CM[0][1]
def specificity(true_negative,false_positive):
    return(true_negative/(true_negative+false_positive))
print('Specifity Score is',specificity(True_negative,False_positive).round(2))

PS=precision_score(Y,Y_pred)
print('Precision Score is',PS.round(2))

fs=f1_score(Y,Y_pred)
print('F1_score is',fs.round(2))

pred_prob=Logreg.predict_proba(X)[:,1]
pred_prob


FPR,TPR,_=roc_curve(Y,pred_prob)
import matplotlib.pyplot as plt
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

auc=roc_auc_score(Y,pred_prob)
print('Area Under Score:',auc.round(2))

