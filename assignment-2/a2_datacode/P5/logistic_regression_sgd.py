#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import random
# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001
# tol = 0.0000000000001

# Step size for gradient descent.
etas = [0.5,0.3,0.1,0.05,0.01]
# etas = [0.3]
# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]
NUM_INPUT = X.shape[0]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]

# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]


# Initialize w.
w = np.array([0.1, 0, 0])

# Error values over all iterations.
e_all = []

DATA_FIG = 1

eall_list = []
for each_eta in etas:
    eall_one_stepsize =[]
    w = np.array([0.1, 0, 0])
    for iter in range(0, max_iter):
        e_per_iter =[]

        for each_input in range(0,NUM_INPUT):
              each_input = np.random.randint(0,NUM_INPUT)
              y = sps.expit(np.dot(X[each_input,:], w))
              grad_e = np.multiply((y - t[each_input]), X[each_input,:].T)
              grad_e = grad_e *(1/NUM_INPUT)

              w_old = w
              w = w - each_eta*grad_e
              e = -( np.multiply(t[each_input], np.log(y)) + np.multiply((1-t[each_input]),np.log(1-y)) )

              e_per_iter.append(e)
        e_avg_per_iter = np.average(e_per_iter)

        print('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e_avg_per_iter, w.T))
        eall_one_stepsize.append(e_avg_per_iter)

        if iter > 0:
            if np.absolute(e_avg_per_iter-eall_one_stepsize[iter-1]) < tol:
                break
    eall_list.append(eall_one_stepsize)

# Plot error over iterations
TRAIN_FIG = 3
plt.figure(TRAIN_FIG, figsize=(8.5, 6))

index = 0
legend_list = []
for each_eall in eall_list:
    plt.plot(each_eall)
    legend_list.append('Step-size: '+str(etas[index]))
    index += 1
plt.legend(legend_list,loc='upper right')
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.grid()
# plt.savefig('sgd_plot_of_neg_log_likelihood_over_epoch.png')
plt.show()
