#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


data = pandas.read_csv('cost_revenue_clean.csv')


# In[3]:


data.describe()


# In[4]:


X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])


# In[5]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[6]:


regression = LinearRegression()
regression.fit(X, y)


# Slope Coefficient:

# In[7]:


regression.coef_ # theta_1 


# In[8]:


# Intercept
regression.intercept_


# In[9]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.plot(X, regression.predict(X), color='red', linewidth=4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[10]:


# revenue = regression.intercept_ + (regression.coef_ * budget)
revenue = regression.intercept_ + (regression.coef_ * 8000000)
print(revenue)


# In[11]:


regression.score(X, y)

