#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
import warnings
warnings.filterwarnings('ignore')


# In[35]:


data=pd.read_csv('data.csv')


# In[36]:


data.shape


# In[37]:


data.head()


# In[38]:


data.isnull().sum()


# In[39]:


data['label'].value_counts()


# In[7]:


print(data['N'].mean())
print(data['P'].mean())
print(data['K'].mean())
print(data['temperature'].mean())
print(data['humidity'].mean())
print(data['ph'].mean())
print(data['rainfall'].mean())


# In[8]:


@interact
def summary(crops=list(data['label'].value_counts().index)):
    x=data[data['label']==crops]
    print('------------------------------------------------------')
    print('statics for nitrogen')
    print('Minimum nitrigen required',x['N'].min())
    print('Minimum nitrigen required',x['N'].mean())
    print('Minimum nitrigen required',x['N'].max())
    print('------------------------------------------------------')
    print('statics for phosphorous')
    print('Minimum phosphorous required',x['P'].min())
    print('Minimum  phosphorous required',x['P'].mean())
    print('Minimum  phosphorousrequired',x['P'].max())
    print('------------------------------------------------------')
    print('statics for potassium')
    print('Minimum potassium required',x['K'].min())
    print('Minimum potassium required',x['K'].mean())
    print('Minimum potassium required',x['K'].max())
    print('------------------------------------------------------')
    print('statics for Temperature')
    print('Minimum Temperature required',x['temperature'].min())
    print('Minimum Temperature required',x['temperature'].mean())
    print('Minimum Temperature required',x['temperature'].max())
    print('------------------------------------------------------')
    print('statics for humidity')
    print('Minimum humidity required',x['humidity'].min())
    print('Minimum humidity required',x['humidity'].mean())
    print('Minimum humidity required',x['humidity'].max()) 
    print('------------------------------------------------------')
    print('statics for ph')
    print('Minimum ph required',x['ph'].min())
    print('Minimum ph required',x['ph'].mean())
    print('Minimum ph required',x['ph'].max()) 
    print('------------------------------------------------------')
    print('statics for rainfall')
    print('Minimum rainfall required',x['rainfall'].min())
    print('Minimum rainfall required',x['rainfall'].mean())
    print('Minimum rainfall required',x['rainfall'].max())
    


# In[9]:


@interact
def compare(conditions=['N','P','K','temperature','humidity','ph','rainfall']):
    print('crop which require greater than avg',conditions)
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print('------------------------------------------')
    print('crop which require less than avg',conditions)
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())


# In[10]:


plt.figure(figsize=(15,7))
plt.subplot(2,4,1)
sns.distplot(data['N'])
plt.xlabel('ratio of Nitrogen',fontsize=12)
plt.grid()
plt.subplot(2,4,2)
sns.distplot(data['P'])
plt.xlabel('ratio of Phosphorous',fontsize=12)
plt.grid()
plt.subplot(2,4,3)
sns.distplot(data['K'])
plt.xlabel('ratio of Potassium',fontsize=12)
plt.grid()
plt.subplot(2,4,4)
sns.distplot(data['temperature'])
plt.xlabel('ratio of temperature',fontsize=12)
plt.grid()
plt.subplot(2,4,5)
sns.distplot(data['humidity'])
plt.xlabel('ratio of humidity',fontsize=12)
plt.grid()
plt.subplot(2,4,6)
sns.distplot(data['ph'])
plt.xlabel('ratio of ph',fontsize=12)
plt.grid()
plt.subplot(2,4,7)
sns.distplot(data['rainfall'])
plt.xlabel('ratio of rainfall',fontsize=12)
plt.grid()
plt.show()


# In[11]:


print('Summer crops')
print(data[(data['temperature']>30)&(data['humidity']>50)]['label'].unique())
print('-----------------------------------------')
print('winter crops')
print(data[(data['temperature']<20)&(data['humidity']>30)]['label'].unique())
print('-----------------------------------------')
print('rainy crops')
print(data[(data['rainfall']>200)&(data['humidity']>30)]['label'].unique())


# In[12]:


from sklearn.cluster import KMeans
x=data.drop(['label'],axis=1)
x=x.values


# In[13]:


x.shape


# In[14]:


y=data['label']
x=data.drop(['label'],axis=1)


# In[15]:


x.shape


# In[16]:


y.shape


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[ ]:





# In[19]:


from sklearn.metrics import confusion_matrix
plt.figure(figsize=(10,10))
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Wistia')
plt.title('confusion matrics for LogisticRegression',fontsize=15)
plt.show()


# In[20]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


# In[44]:


prediction=model.predict((np.array([[90,40,40,20,80,7,200]])))
print('the suggested crop is :',prediction)


# In[45]:


from sklearn.tree import DecisionTreeClassifier 
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[46]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


# In[ ]:




