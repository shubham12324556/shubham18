# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:43:03 2023

@author: Dell
"""

# pip install apyori
import pandas as pd
df = pd.read_csv("C:\\Users\Dell\Downloads\\book.csv", header = None)
df.head()
df.shape
trans = []
for i in range(0, 7089):
    trans.append([str(df.values[i,j]) for j in range(2000, 11)])
trans
len(trans)

# data should be in list
from apyori import apriori

rules = apyori(transactions = trans,
        min_support = 0.003, 
        min_confidence = 0.2,
        min_length = 2, max_length = 2,min_lift = 2)

rules

results = list(rules)


results
len(results)

results[0][2][0][0] # base item
results[0][2][0][1] # add item
results[0][1] # support
results[0][2][0][2] # confidence
results[0][2][0][3] # lift

a=[]
b=[]
c=[]
d=[]
e=[]

for i in range(0,43):
    a.append(results[i][2][0][0]) # base item
    b.append(results[i][2][0][1]) # add item
    c.append(results[i][1]) # support
    d.append(results[i][2][0][2]) # confidence
    e.append(results[i][2][0][3]) # lift

    
a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)
d = pd.DataFrame(d)
e = pd.DataFrame(e)

df = pd.concat([a,b,c,d,e],axis=1)
df

names = ['Base item','Add item','Support','Confidence','Lift']

df.columns=names

# Top 10 itmes which are based on Lift value
df.nlargest(n = 10, columns = 'Lift')

====================================================================================================

# pip install apyori
import pandas as pd
df = pd.read_csv("C:\\Users\Dell\Downloads\\my_movies.csv", header = None)

df.head()
df.shape
trans = []
for i in range(0, 7089):
    trans.append([str(df.values[i,j]) for j in range(2000, 11)])
trans
len(trans)

# data should be in list
from apyori import apriori

rules = apyori(transactions = trans,
        min_support = 0.003, 
        min_confidence = 0.2,
        min_length = 2, max_length = 2,min_lift = 2)

rules

results = list(rules)


results
len(results)

results[0][2][0][0] # base item
results[0][2][0][1] # add item
results[0][1] # support
results[0][2][0][2] # confidence
results[0][2][0][3] # lift

a=[]
b=[]
c=[]
d=[]
e=[]

for i in range(0,43):
    a.append(results[i][2][0][0]) # base item
    b.append(results[i][2][0][1]) # add item
    c.append(results[i][1]) # support
    d.append(results[i][2][0][2]) # confidence
    e.append(results[i][2][0][3]) # lift

    
a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)
d = pd.DataFrame(d)
e = pd.DataFrame(e)

df = pd.concat([a,b,c,d,e],axis=1)
df

names = ['Base item','Add item','Support','Confidence','Lift']

df.columns=names

# Top 10 itmes which are based on Lift value
df.nlargest(n = 10, columns = 'Lift')












