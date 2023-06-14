#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


dataset = pd.read_csv('C:/Users/Mehmet Alpay/Desktop/Software/anaconda kodlama/applerevenue .csv')
dataset = shuffle(dataset)
dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# In[7]:


dataset.isnull().sum()


# In[8]:


sns.set()
sns.displot(data=dataset, x='open', kde=True, linewidth=1) 
plt.title("Analysing starting rate of the stock")
plt.show()


# In[9]:


sns.displot(data=dataset, x='high', kde=True, linewidth=1)
plt.title('Analysing highest rate of the stock for the day')
plt.show()


# In[10]:


sns.displot(data=dataset, x='low', kde=True, linewidth=1)
plt.title('Analysing lowest rate of the stock for the day')
plt.show()


# In[11]:


sns.displot(data=dataset, x='close', kde=True, linewidth=1)
plt.title('Analysing closing rate of stock for the day')
plt.show()


# In[12]:


sns.displot(data=dataset, x='volume', kde=True, linewidth=1)
plt.title('Analysing volume for the day')
plt.show()


# In[13]:


X = dataset.drop(columns=['date', 'profit or not', 'volume'])
Y = dataset['profit or not']
X.head()


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[15]:


training_model = LogisticRegression()
training_model.fit(x_train, y_train)
y_train_prediction = training_model.predict(x_train)
y_train_prediction


# In[16]:


training_model.score(x_train, y_train)


# In[17]:


training_cf_matrix = confusion_matrix(y_train, y_train_prediction)
training_cf_matrix


# In[18]:


sns.heatmap(training_cf_matrix/np.sum(training_cf_matrix), annot=True, fmt='.2%')
plt.title('Analysis of model performance for training dataset')
plt.show()


# In[19]:


y_test_prediction = training_model.predict(x_test)
y_test_prediction


# In[20]:


training_model.score(x_test, y_test)


# In[21]:


testing_cf_matrix = confusion_matrix(y_test, y_test_prediction)
testing_cf_matrix


# In[22]:


sns.heatmap(testing_cf_matrix/np.sum(testing_cf_matrix), annot=True, fmt='0.2%')
plt.title('Analysing performance of the model for testing dataset')
plt.show()


# In[23]:


dataset[500:505]


# In[24]:


trial_input = dataset.drop(columns=['date', 'volume', 'profit or not'])
true_positive_input = np.asarray([[0.1081, 0.1081, 0.1007, 0.1020]])
true_positive_input


# In[25]:


training_model.predict(true_positive_input)


# In[26]:


dataset[35:40]


# In[27]:


true_negative_input = np.asarray([[0.0998, 0.1007, 0.0998, 0.0998]])
true_negative_input


# In[28]:


training_model.predict(true_negative_input)


# In[ ]:




