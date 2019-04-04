#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
base = pd.read_csv('house-prices.csv')
base.head()


# In[4]:


base.shape


# In[5]:


X = base.iloc[:, 5:6].values
y = base.iloc[:, 2:3].values


# In[6]:


X


# In[7]:


y


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


# In[9]:


X


# In[10]:


y


# In[11]:


import tensorflow as tf


# In[12]:


colunas = [tf.feature_column.numeric_column('x', shape = [1])]


# In[13]:


colunas


# In[14]:


regressor = tf.estimator.LinearRegressor(feature_columns=colunas)


# In[15]:


from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)


# In[16]:


X_treinamento.shape


# In[17]:


y_treinamento.shape


# In[18]:


X_teste.shape


# In[19]:


y_teste.shape


# In[20]:


funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x': X_treinamento}, y_treinamento,
                                                        batch_size = 32, num_epochs = None, 
                                                        shuffle = True)


# In[21]:


funcao_teste = tf.estimator.inputs.numpy_input_fn({'x': X_teste}, y_teste, batch_size = 32, 
                                                  num_epochs = 1000, shuffle = False)


# In[22]:


regressor.train(input_fn = funcao_treinamento, steps = 10000)


# In[23]:


metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)


# In[24]:


metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)


# In[25]:


metricas_treinamento


# In[26]:


metricas_teste


# In[27]:


import numpy as np
novas_casas = np.array([[800], [900], [1000]])
novas_casas


# In[28]:


novas_casas = scaler_x.transform(novas_casas)
novas_casas


# In[29]:


funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas}, shuffle = False)


# In[30]:


previsoes = regressor.predict(input_fn = funcao_previsao)


# In[31]:


previsoes


# In[32]:


list(previsoes)


# In[33]:


for p in regressor.predict(input_fn = funcao_previsao):
    #print(p['predictions'])
    print(scaler_y.inverse_transform(p['predictions']))


# In[ ]:




