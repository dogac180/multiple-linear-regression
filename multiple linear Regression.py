#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("50_Startups.csv")


# In[5]:


x = data.iloc[:, 0].values
y = data.iloc[:, -1].values


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[7]:


x_train


# In[8]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)


# In[9]:


y_pre = regressor.predict(x_test.reshape(-1,1))


# In[10]:


plt.scatter(x_train,y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train.reshape(-1,1)), color="blue")
plt.title("r&d vs profit")
plt.xlabel("r%d")
plt.ylabel("profit")
plt.show()


# In[11]:


plt.scatter(x_test,y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train.reshape(-1,1)), color="blue")
plt.title("r&d vs profit")
plt.xlabel("r%d")
plt.ylabel("profit")
plt.show()


# In[12]:


x1 = data.iloc[:, 1].values
y1 = data.iloc[:, -1].values


# In[13]:


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)


# In[14]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x1_train.reshape(-1,1), y1_train)


# In[15]:


y_predict = reg.predict(x1_test.reshape(-1,1))


# In[16]:


plt.scatter(x1_train, y1_train, color = "red")
plt.plot(x1_train, reg.predict(x1_train.reshape(-1,1)), color = "blue")
plt.title("admin expense vs profit")
plt.xlabel("admin expense")
plt.ylabel("profit")
plt.show()


# # Multiple Linear Regression

# In[17]:


X = data.iloc[:, :4].values
Y = data.iloc[:, -1].values


# In[18]:


#encode the dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
col_transfor = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [3])],remainder = "passthrough")
X = np.array(col_transfor.fit_transform(X))


# In[19]:


X


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[21]:


from sklearn.linear_model import LinearRegression
regresso = LinearRegression()
regresso.fit(X_train, Y_train)


# In[ ]:




