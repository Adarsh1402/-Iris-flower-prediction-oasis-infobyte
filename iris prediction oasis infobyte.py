#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')


# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


get_ipython().system('pip install matplotlib')


# In[66]:


get_ipython().system('pip install seaborn')


# In[67]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


#for reading the csv file
df = pd.read_csv('Iris.csv')
df.head()


# In[69]:


#for deleting the ID column
df = df.drop(columns=['Id'])
df.head()


# In[70]:


#to display stats of data
df.describe()


# In[71]:


df.info()


# In[72]:


#to display no of samples on each class
df['Species'].value_counts()


# In[73]:


#processing of dataset

#null values
df.isnull().sum()


# In[19]:


#visualisation in histogram
df['SepalLengthCm'].hist()


# In[20]:


df['SepalWidthCm'].hist()


# In[21]:


df['PetalLengthCm'].hist()


# In[22]:


df['PetalWidthCm'].hist()


# In[23]:


#scatterplot

colors =['red','yellow','blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[28]:


for i in range(3):
    x = df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[29]:


for i in range(3):
    x = df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[30]:


for i in range(3):
    x = df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[31]:


for i in range(3):
    x = df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[74]:


#correlation matrix
df.corr()


# In[36]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True , ax=ax , cmap='coolwarm')


# In[37]:


#lable encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[39]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[57]:


#model training
from sklearn.model_selection import train_test_split

#train - 70
#test - 30
X = df.drop(columns = ['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30)


# In[58]:


#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[59]:


model.fit(x_train,y_train)


# In[61]:


#prtinting metric for performance
print('Accuracy:' , model.score(x_test, y_test)*100)


# In[62]:


#knn model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[63]:


model.fit(x_train, y_train)


# In[64]:


#prtinting metric for performance
print('Accuracy:' , model.score(x_test, y_test)*100)


# In[65]:


#decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[55]:


model.fit(x_train, y_train)


# In[56]:


#prtinting metric for performance
print('Accuracy:' , model.score(x_test, y_test)*100)


# In[ ]:




