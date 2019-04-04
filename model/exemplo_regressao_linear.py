# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:57:52 2019

@author: Renan F Rodrigues
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

x_pred = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
#x_pred = np.array([18, 23, 28, 33, 38, 43, 48, 53, 58, 63])
y_targ = np.array([[871], [1132], [1042 ], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])
#y_targ = np.array([871, 1132, 1042, 1356, 1488, 1638, 1569, 1754, 1866, 1900])
plt.scatter(x_pred, y_targ)
regressor = LinearRegression()
regressor.fit(x_pred, y_targ)
print ('b0 ' + str(regressor.intercept_))
print ('b1 ' + str(regressor.coef_))
print('Example of manual prediction ' + str(regressor.intercept_ + regressor.coef_ * 40))
print('Prediction for specific case using the library ' + str(regressor.predict([[40]])))
prevs = regressor.predict(x_pred)
print('Previsions based on features ' + str(prevs))
error_tax = mean_absolute_error(y_targ, prevs)
print('Error tax ' + str(error_tax))
plt.plot(x_pred, y_targ, 'o')
plt.plot(x_pred, prevs, color = 'red')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')

