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

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
START_FEATURE_INDEX = 8
format =False
with_bias = True
basis = 'sigmoid'
feature = 11
mu = [100,10000]
s= [2000.0,2000.0]

x_train = x[0:N_TRAIN,feature-START_FEATURE_INDEX]
x_test = x[N_TRAIN:,feature-START_FEATURE_INDEX]
(w, tr_err) = a1.linear_regression(x_train,t_train,basis=basis,with_bias=with_bias,mu=mu,s=s)
print("w shape: ",w.shape)
(t_est, te_err) = a1.evaluate_regression(x_test,t_test,w,basis=basis,with_bias=with_bias,mu=mu,s=s)
if format:
    formatted_tr_err=np.array2string(tr_err,formatter={'float_kind':'{0:.10f}'.format})
    formatted_te_err=np.array2string(te_err,formatter={'float_kind':'{0:.10f}'.format})
    print("train error: ", formatted_tr_err," + test_error: ",formatted_te_err)
else:
    print("train error: ", tr_err," + test_error: ",te_err)
print('\n')

# Produce a plot of results.
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
# x_ev = np.linspace(np.asscalar(min(min(x_train),min(x_test))), np.asscalar(max(max(x_train),max(x_test))), num=500)

# TO DO::
# Perform regression on the linspace samples.
# Put your regression estimate here in place of y_ev.
(t_ev, te_err) = a1.evaluate_regression(np.asmatrix(x_ev).T,t_test,w,basis=basis,with_bias=with_bias,mu=mu,s=s)
plt.plot(x_train,t_train,'go')
plt.plot(x_test,t_test,'bo')
plt.plot(x_ev,t_ev,'r.-')
plt.xlabel('Train error:'+str(np.asscalar(tr_err))+", Test error:"+str(np.asscalar(te_err)))
print("Plotting: train error: ", tr_err," + test_error: ",te_err)

plt.legend(['Train Data','Test Data','Sigmoid Basis Function'])
plt.title('Visualization of a function and some data points \nFeature No.11(GNI)')
# pylab.savefig('visualize_sigmoid_feature11',bbox_inches='tight')
plt.show()
plt.clf()
