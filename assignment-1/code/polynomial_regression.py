#!/usr/bin/env python
'''Name: Nattapat Juthaprachakul, Student ID: 301350117'''

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pylab
from pylab import *
# np.set_printoptions(precision=10, suppress=True)
(countries, features, values) = a1.load_unicef_data()
targets = values[:,1] #col2 (under-5 mortality rate 2011)
x = values[:,7:] #col 8-40
# x = a1.normalize_data(x)

N_TRAIN = 100   #countries 1-100 Afghan->Luxem
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

'''############## 5.1 Getting started #################'''
df = pd.DataFrame(values,index=countries,columns=features)

'''5.1.1 Which country had the highest child mortality rate in 1990? What was the rate?'''
print(df.iloc[:,0].idxmax())  #ANS: Niger
'''5.1.2 Which country had the highest child mortality rate in 2011? What was the rate?'''
print(df.iloc[:,1].idxmax())  #ANS: Sierra Leon
'''5.1.3 Some countries are missing some features (see original .xlsx/.csv spreadsheet). How is this
handled in the function assignment1.load unicef data()?'''
print('\n')
''' ###################################################'''


# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
with_bias = True
basis = 'polynomial'
format =False
list_degree= [1,2,3,4,5,6]
# list_degree = [3]
train_err = {}
test_err = {}
for degree in list_degree:
    (w, tr_err) = a1.linear_regression(x_train,t_train,basis=basis,degree=degree,with_bias=with_bias)
    print("degree=",degree," and w shape: ",w.shape)
    (t_est, te_err) = a1.evaluate_regression(x_test,t_test,w,basis=basis,degree=degree,with_bias=with_bias)
    train_err[degree] = np.asscalar(tr_err)
    test_err[degree] = np.asscalar(te_err)
    if format:
        formatted_tr_err=np.array2string(tr_err,formatter={'float_kind':'{0:.10f}'.format})
        formatted_te_err=np.array2string(te_err,formatter={'float_kind':'{0:.10f}'.format})
        print("train error: ", formatted_tr_err," + test_error: ",formatted_te_err)
    else:
        print("train error: ", tr_err," + test_error: ",te_err)
    print('\n')
print("train err dict: ",train_err)
print("test err dict: ",test_err)

# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.grid()
# pylab.savefig('polynomial_regression',bbox_inches='tight')
plt.show()
