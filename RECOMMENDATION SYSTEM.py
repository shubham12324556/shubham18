# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:05:39 2023

@author: Dell
"""

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\Dell\Downloads\\book.csv')
df.shape
df.head()

df.shape
import warnings
warnings.filterwarnings('ignore')
df.sort_values('User.ID')
len(df)
len(df.userId.unique())

len(df.movie.unique())

df['rating'].value_counts()
df['rating'].hist()

user_df = df.pivot(index='userId',
                                 columns='movie',
                                 values='rating')

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)

user_df

# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,
                                  metric='cosine')


user_sim.shape

user_sim_df = pd.DataFrame(user_sim)
user_sim_df

user_sim_df.index   = df.userId.unique()
user_sim_df.columns = df.userId.unique()

user_sim_df.head()
user_sim_df.shape


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:7, 0:7]


user_sim_df.idxmax(axis=1)[0:10]

df[(df['userId']==3) | (df['userId']==11)]

df[(df['userId']==6) | (df['userId']==168)]

