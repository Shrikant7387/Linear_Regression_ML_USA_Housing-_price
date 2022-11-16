#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r'C:\Users\Admin\Downloads\Data thecnogeeks\Data_Engineering\DataScience\1. Linear-Regression\USA_Housing.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.corr()


# In[6]:


sns.jointplot(x='Price',y='Avg. Area Number of Rooms',data=data,kind='reg')


# In[7]:


sns.pairplot(data)


# In[9]:


data.columns


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)


# In[13]:


x_train.shape


# In[14]:


y_train.shape


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lgr = LinearRegression()


# In[17]:


lgr.fit(x_train,y_train)


# In[18]:


y_pred=lgr.predict(x_test)


# In[20]:


lgr.coef_


# In[21]:


y_pred[102]


# In[24]:


coeff_df = pd.DataFrame(lgr.coef_,x.columns,columns=['Coefficient'])
coeff_df


# In[26]:


x.mean()


# y_test.iloc[102]

# In[27]:


y_pred[100]


# In[28]:


y_test.iloc[100]


# In[29]:


plt.scatter(y_test,y_pred)


# In[31]:


plt.scatter(y_test,y_test)


# In[34]:


from sklearn import metrics


# In[35]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred))


# In[36]:


print('MSE:',metrics.mean_squared_error(y_test,y_pred))


# In[37]:


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




