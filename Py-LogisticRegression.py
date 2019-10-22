#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# import data
get_ipython().system('wget -O ChurnData.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv')
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()


# In[4]:


# Define dataset
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()


# In[6]:


# define x nd y of dataset
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[7]:


y = np.asarray(churn_df['churn'])
y [0:5]


# In[8]:


# Standardize data to zero mean
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[9]:


# test train data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[10]:


# Do regresion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[11]:


# Predict
yhat = LR.predict(X_test)
yhat


# In[12]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[13]:


# Model evaluation ising jaccard index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[14]:


# Model evaluayion using log loss
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[ ]:




