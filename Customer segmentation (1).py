#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df =pd.read_csv("Mall_Customers.csv")
df.head(10)


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


X=df.iloc[:,[3,4]].values


# In[9]:


from sklearn.cluster import KMeans
wcss=[]


# In[21]:


for i in range(1,11):
    kmeans=KMeans(n_clusters= i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[22]:


plt.plot(range(1,11),wcss)
plt.title("The elobow method")
plt.xlabel("No. of clusters")
plt.ylabel('WCSS Values')
plt.show()


# In[45]:


kmeanmodel=KMeans(n_clusters = 5, init ='k-means++',random_state=0)


# In[46]:


y_kmeans=kmeanmodel.fit_predict(X)


# In[48]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=80, c="red",label="Customer 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=80, c="blue",label="Customer 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=80, c="yellow",label="Customer 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=80, c="cyan",label="Customer 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s=80, c="black",label="Customer 5")
plt.title('Clusters of customers')
plt.xlabel("Annual Income")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




