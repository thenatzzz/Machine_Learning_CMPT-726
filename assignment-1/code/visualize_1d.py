#!/usr/bin/env python
'''Name: Nattapat Juthaprachakul, Student ID: 301350117'''


import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *

(countries, features, values) = a1.load_unicef_data()
targets = values[:,1] #col2 (under-5 mortality rate 2011)
x = values[:,7:] #col 8-40
# x = a1.normalize_data(x)

N_TRAIN = 100   #countries 1-100 Afghan->Luxem
START_FEATURE_INDEX=8
END_FEATURE_INDEX=15

def plot_1d(feature):
    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]

    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    # x_ev = np.linspace(np.asscalar(min(min(x_train),min(x_test))), np.asscalar(max(max(x_train),max(x_test))), num=500)

    # TO DO::
    # Perform regression on the linspace samples.
    # Put your regression estimate here in place of y_ev.

    basis = 'polynomial'
    degree = 3
    with_bias = True
    (w, tr_err) = a1.linear_regression(x_train,t_train,basis=basis,degree=degree,with_bias=with_bias)
    (t_ev, te_err) = a1.evaluate_regression(np.asmatrix(x_ev).T,t_test,w,basis=basis,degree=degree,with_bias=with_bias)

    plt.plot(x_train,t_train,'go')
    plt.plot(x_test,t_test,'bo')
    plt.plot(x_ev,t_ev,'r.-')

    plt.legend(['Train Data','Test Data','Polynomial Degree '+str(degree)])
    plt.title('Visualization of a function and some data points \nFeature No.'+ str(feature+START_FEATURE_INDEX))
    # pylab.savefig('visualize_1d_feature'+str(feature+START_FEATURE_INDEX),bbox_inches='tight')
    plt.show()
    plt.clf()

# Feature [8,9,10,11,12,13,14,15] == Index in values [0,1,2,3,4,5,6,7]
dict_feature = {}
index_feature = 0
for no_feature in range(START_FEATURE_INDEX,END_FEATURE_INDEX+1):
    dict_feature[no_feature] = index_feature
    index_feature += 1

plot_1d(dict_feature[11])
plot_1d(dict_feature[12])
plot_1d(dict_feature[13])
