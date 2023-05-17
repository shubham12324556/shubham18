# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:55:28 2023

@author: Dell
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('C:\\Users\Dell\Downloads\\wine.csv')
df

df.shape

df.head()

from sklearn.decomposition import PCA

pca=PCA()
df1=pd.DataFrame(pca.fit_transform(df))
df1.columns=list(df)
df1

t1 = pca.explained_variance_ratio_

t1[0]

t1[1]

t1[2]





X=df1.iloc[:,:3]
X

get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D

fig1=plt.figure()
ax=Axes3D(fig1)
ax.scatter(X['Type'],X['Alcohol'],X['Malic'])
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster1=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
Y1=cluster1.fit_predict(X)
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()

plt.figure(figsize=(10,7))
plt.scatter(X['Type'],X['Alcohol'],c=cluster1.labels_,cmap='rainbow')

cluster2=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
Y2=cluster2.fit_predict(X)
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()

plt.figure(figsize=(10,7))
plt.scatter(X['Type'],X['Malic'],c=cluster1.labels_,cmap='rainbow')

Y_clust1=pd.DataFrame(Y1)
Y_clust1[0].value_counts()

Y_clust2=pd.DataFrame(Y2)
Y_clust2[0].value_counts()

from sklearn.cluster import KMeans

l1=[]
for i in range(1,17):
    kmeans=KMeans(n_clusters=i)
    kmeans=kmeans.fit(X)
    l1.append(kmeans.inertia_)
print(l1)

import matplotlib.pyplot as plt

plt.scatter(range(1,17),l1)
plt.plot(range(1,17),l1,color='red')
plt.show()

from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X

from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit(SS_X)

Y1=dbscan.labels_
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()

df1=pd.concat([X,Y_new1],axis=1)
df1

df1[df1[0]==-1]

df_new1=df1[df1[0]!=-1]
df_new1.shape

Y2=dbscan.labels_
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()

df2=pd.concat([X,Y_new1],axis=1)
df2

df2[df2[0]==-1]

df_new2=df2[df2[0]!=-1]
df_new2.shape


