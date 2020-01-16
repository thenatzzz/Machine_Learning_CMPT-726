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
NUM_FEATURE = 8
list_feature = list(range(NUM_FEATURE))

x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

basis = 'polynomial'
format =False
degree = 3
train_err = {}
test_err = {}
with_bias = True #apply bias or not
for each_feature in list_feature:
    x_train = x[0:N_TRAIN,each_feature]
    x_test = x[N_TRAIN:,each_feature]
    (w, tr_err) = a1.linear_regression(x_train,t_train,basis=basis,degree=degree,with_bias=with_bias)
    (t_est, te_err) = a1.evaluate_regression(x_test,t_test,w,basis=basis,degree=degree,with_bias=with_bias)
    train_err[each_feature] = np.asscalar(tr_err)
    test_err[each_feature] = np.asscalar(te_err)
    if format:
        formatted_tr_err=np.array2string(tr_err,formatter={'float_kind':'{0:.10f}'.format})
        formatted_te_err=np.array2string(te_err,formatter={'float_kind':'{0:.10f}'.format})
        print("train error: ", formatted_tr_err," + test_error: ",formatted_te_err)
    else:
        print("train error: ", tr_err," + test_error: ",te_err)
    print('\n')
print("train err dict: ",train_err)
print("test err dict: ",test_err)

labels = np.array(list_feature)+NUM_FEATURE
list_train_err = list(train_err.values())
list_test_err = list(test_err.values())

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, list_train_err, width, label='Train Error')
rects2 = ax.bar(x + width/2, list_test_err, width, label='Test Error')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS')
ax.set_xlabel('Features')
if with_bias:
    bias = "with Bias"
else:
    bias= "without Bias"
ax.set_title('Polynomial Degee 3 \nRMS of train/test error by features '+bias)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
# pylab.savefig('polynomial_regression_1d',bbox_inches='tight')
plt.show()
