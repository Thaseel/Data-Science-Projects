#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


dataset = pd.read_csv("fraud_dataset.csv")


# In[3]:


dataset.head(10)


# In[7]:


sns.countplot(x="Fraud_Risk",data=dataset)


# In[8]:


sns.countplot(x="Fraud_Risk",hue="Gender",data=dataset)


# In[9]:


dataset.isnull()


# In[10]:


dataset.isnull().sum()


# In[14]:


sns.heatmap(dataset.isnull(),yticklabels=False)


# In[15]:


sns.boxplot(x="Fraud_Risk",y="Gender",data=dataset)


# In[16]:


dataset.head(2)


# In[19]:


dataset.drop(['Gender','Married','Dependents','LoanAmount','Loan_Term','Locality'],axis=1,inplace=True)


# In[20]:


dataset.head(2)


# In[22]:


X = dataset.drop("Fraud_Risk",axis=1)
y = dataset["Fraud_Risk"]


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


logmodel = LogisticRegression()


# In[29]:


logmodel.fit(X_train,y_train)


# In[30]:


predictions = logmodel.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


classification_report(y_test,predictions)


# In[33]:


from sklearn.metrics import confusion_matrix


# In[34]:


confusion_matrix(y_test,predictions)


# In[35]:


from sklearn.metrics import accuracy_score


# In[38]:


df = accuracy_score(y_test,predictions)*100


# In[39]:


display(df)


# In[ ]:




