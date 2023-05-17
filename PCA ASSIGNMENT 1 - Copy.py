# -*- coding: utf-8 -*-
"""
Created on Sun May  7 09:34:31 2023

@author: Dell
"""
QUESTION 1

import pandas as pd  
customer_data = pd.read_csv('C:\\Users\Dell\Downloads\\crime_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 2:5].values 
X.shape


##############################################################################
%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,1], X[:,2], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()
##############################################################################

# Initializing KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)

Y = pd.DataFrame(labels)
Y[0].value_counts()


kmeans.inertia_

l1= []
for i in range(1,13):
    kmeans = KMeans(n_clusters=i)
    kmeans = kmeans.fit(X)
    l1.append((kmeans.inertia_))
    
    
print(l1)
    
# elbow plot, scree plot
import matplotlib.pyplot as plt
plt.scatter(range(1,13),l1)
plt.plot(range(1,13),l1,color='red')
plt.show()


plt.figure(figsize=(10, 7))  
plt.scatter(X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')  

#====================================================================
# DBSCAN  ---------------> 
#====================================================================

import pandas as pd  
customer_data = pd.read_csv('C:\Users\Dell\Downloads\\crime_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 2:5].values 
X.shape

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

Y = dbscan.labels_
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

df = pd.concat([customer_data,Y_new], axis=1)
df
df[df[0] == -1 ]

df_new = df[df[0] != -1 ]
df_new.shape


==================================================================================================================
import pandas as pd  
 
 
customer_data = pd.read_excel('C:\\Users\Dell\Downloads\\EastWestAirlines.xlsx', sheet_name=1) 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 2:5].values 
X.shape


##############################################################################
%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,1], X[:,2], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()
##############################################################################

# Initializing KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)

Y = pd.DataFrame(labels)
Y[0].value_counts()


kmeans.inertia_

l1= []
for i in range(1,13):
    kmeans = KMeans(n_clusters=i)
    kmeans = kmeans.fit(X)
    l1.append((kmeans.inertia_))
    
    
print(l1)
    
# elbow plot, scree plot
import matplotlib.pyplot as plt
plt.scatter(range(1,13),l1)
plt.plot(range(1,13),l1,color='red')
plt.show()


plt.figure(figsize=(10, 7))  
plt.scatter(X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')  

#====================================================================
# DBSCAN  ---------------> 
#====================================================================

import pandas as pd  
customer_data = pd.read_excel('C:\\Users\Dell\Downloads\\EastWestAirlines.xlsx', sheet_name=1) 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 3:5].values 
X.shape

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

Y = dbscan.labels_
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

df = pd.concat([customer_data,Y_new], axis=1)
df
df[df[0] == -1 ]

df_new = df[df[0] != -1 ]
df_new.shape
==========================================================================================================

