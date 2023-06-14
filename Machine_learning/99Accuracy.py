#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import warnings 
warnings.filterwarnings("ignore")


# In[2]:


dataset = pd.read_csv('C:/Users/Mehmet Alpay/Desktop/Software/anaconda kodlama/applerevenue .csv')
dataset.head()


# In[3]:


dataset.describe()


# In[4]:


X = dataset.drop(columns=['date', 'profit or not', 'volume'])
Y = dataset['profit or not']
X.head()


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.35, random_state=42)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[6]:


training_model = LogisticRegression(class_weight='balanced')
training_model.fit(x_train, y_train)
y_train_prediction = training_model.predict(x_train)
y_train_prediction


# In[7]:


training_model.score(x_train, y_train)


# In[8]:


training_cf_matrix = confusion_matrix(y_train, y_train_prediction)

training_cf_matrix


# In[9]:


b = pd.DataFrame(training_cf_matrix)
b.rename(columns = {0: 'True', 1:'False'}, inplace = True)
b


# In[11]:


sns.heatmap(training_cf_matrix/np.sum(training_cf_matrix), annot=True, fmt='.2%')
plt.title('Analysis of model performance for training dataset')
plt.show()


# In[12]:


y_test_prediction = training_model.predict(x_test)
y_test_prediction


# In[13]:


training_model.score(x_test, y_test)


# In[14]:


testing_cf_matrix = confusion_matrix(y_test, y_test_prediction)
testing_cf_matrix


# In[15]:


a = pd.DataFrame(testing_cf_matrix)
a.rename(columns = {0: 'True', 1:'False'}, inplace = True)
a


# In[16]:


sns.heatmap(testing_cf_matrix/np.sum(testing_cf_matrix), annot=True, fmt='0.2%')
plt.title('Analysing performance of the model for testing dataset')
plt.show()


# In[ ]:




