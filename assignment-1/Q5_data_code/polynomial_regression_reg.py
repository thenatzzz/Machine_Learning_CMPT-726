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
x = a1.normalize_data(x)

N_TRAIN = 100   #countries 1-100 Afghan->Luxem
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
with_bias = True
basis = 'polynomial'
list_reg_lambda =[0,0.01,0.1,1,10,100,1000,10000]
degree = 2
train_err = {}
test_err = {}
MAX_FOLD = 10
def apply_crossvalidation(x_train,t_train,which_fold,max_fold):
    total_input = x_train.shape[0]
    # num_xt_train ->majority fold or 9 fold
    # num_xt_validation -> 1 fold
    num_minority_fold = total_input // max_fold
    num_majority_fold= total_input - num_minority_fold

    start_index_minority = (which_fold-1)*num_minority_fold
    end_index_minority  = start_index_minority + num_minority_fold

    minority_x_train, minority_t_train=np.empty,np.empty
    majority_x_train, majority_t_train=np.empty, np.empty
    count = 0
    for i in range(0,total_input):
        if(i >= start_index_minority and i< end_index_minority):
            minority_x_train=x_train[start_index_minority:end_index_minority,:]
            minority_t_train= t_train[start_index_minority:end_index_minority,:]
            count += 1
        else:
            if count == i:
                majority_x_train,majority_t_train = x_train[i,:],t_train[i,:]
            majority_x_train = np.concatenate((majority_x_train,x_train[i,:]), axis=0) # join matrix
            majority_t_train = np.concatenate((majority_t_train,t_train[i,:]), axis=0) # join matrix
    return majority_x_train,majority_t_train,minority_x_train,minority_t_train

original_x_train = x_train
original_t_train = t_train
collection_fold_lambda = {}
for which_fold in range(1,MAX_FOLD+1):
    x_train,t_train,x_validation,t_validation= apply_crossvalidation(original_x_train,original_t_train,which_fold,MAX_FOLD)
    list_val_err = []
    for reg_lambda in list_reg_lambda:
        (w, tr_err) = a1.linear_regression(x_train,t_train,reg_lambda=reg_lambda,basis=basis,degree=degree,with_bias=with_bias)
        print("degree=",degree," and w shape: ",w.shape)
        (val_est, val_err) = a1.evaluate_regression(x_validation,t_validation,w,reg_lambda=reg_lambda,basis=basis,degree=degree,with_bias=with_bias)
        print("reg lambda:",reg_lambda, " -- val_err:",val_err)
        list_val_err.append(np.asscalar(val_err))
    collection_fold_lambda['Fold '+str(which_fold)] =list_val_err

count = 0
for key,value in collection_fold_lambda.items():
    if count ==0:
        np_fold_lambda = np.array(collection_fold_lambda['Fold 1'])
    else:
        np_fold_lambda=np.vstack((np_fold_lambda,np.array(value)))
    count = count+ 1
np_lambda_fold = np_fold_lambda.T
mean_val_err_per_lambda = np.mean(np_lambda_fold,axis=1)
print("Val error: ",mean_val_err_per_lambda)
print("List of lambda: ",list_reg_lambda)

plt.semilogx( list_reg_lambda,mean_val_err_per_lambda)
plt.ylabel("Validation Error")
plt.xlabel("Semi log Lambda")
plt.title('Polynomial Degee '+str(degree)+' \nRMS of validation set by different lambda')
plt.grid()
#pylab.savefig('polynomial_regression_reg',bbox_inches='tight')
plt.show()
