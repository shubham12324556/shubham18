# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:47:47 2023

@author: Dell
"""

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\Dell\Downloads\\book.csv',encoding ='iso-8859-1')

df.shape
df.head()



df.sort_values('User.ID')
len(df)
len(df.userId.unique())

len(df.movie.unique())

df['rating'].value_counts()
df['rating'].hist()

User_df = df.pivot(index='User.ID',
                                 columns='movie',
                                 values='rating')

#Impute those NaNs with 0 values
User_df.fillna(0, inplace=True)

User_df

# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
User_sim = 1 - pairwise_distances(User_df.values,
                                  metric='cosine')


User_sim.shape

User_sim_df = pd.DataFrame(User_sim)
User_sim_df

User_sim_df.index   = df.UserId.unique()
User_sim_df.columns = df.UserId.unique()

User_sim_df.head()
User_sim_df.shape


np.fill_diagonal(User_sim, 0)
User_sim_df.iloc[0:7, 0:7]


User_sim_df.idxmax(axis=1)[0:10]

df[(df['User.ID']==3) | (df['User.ID']==11)]

df[(df['User.ID']==6) | (df['User.ID']==168)]








