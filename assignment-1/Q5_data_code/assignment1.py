'''Name: Nattapat Juthaprachakul, Student ID: 301350117'''
"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec)/stdvec



def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=[0], s=[1],with_bias=True):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(x,degree,basis,with_bias,mu,s)

    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        lambda_identity_matrix = reg_lambda*np.identity(phi.shape[1])
        first_term= np.linalg.pinv(np.dot(phi.T,phi)+lambda_identity_matrix)
        second_term = np.dot(phi.T,t)
        w = np.dot(first_term,second_term)
    else:
        # no regularization
        # w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T,phi)),phi.T),t)
        w = np.linalg.pinv(phi)*t

    # Measure root mean squared error on training data.
    sum_error = 0.0
    for each_sample in range(0,x.shape[0]):
        single_error = t[each_sample]-np.dot(w.T,phi[each_sample,:].T)
        single_error_squared = single_error**2
        sum_error = sum_error + single_error_squared
    total_error = 0.5*sum_error
    train_err = np.sqrt(2.0*total_error/x.shape[0])

    return (w, train_err)



def design_matrix(input,degree,basis=None,with_bias=True,mu=[0],s=[1]):
    """ Compute a design matrix Phi from given input datapoints and basis.
    Args:
        input, degree= polynomial degree, basis = name of basis used
    Returns:
      phi design matrix
    """
    if basis == 'polynomial':
        original_phi = input
        phi = input
        for each_degree in range(1,degree+1):
            if degree==1: # if polynomial degree ==1 does nothing
                break
            if each_degree == 1:
                continue
            temp_phi = np.power(original_phi,each_degree)
            phi = np.concatenate((phi,temp_phi), axis=1) # join matrix
    elif basis == 'sigmoid':
        original_phi = input
        phi = input
        count =  0
        for single_mu,single_s in mu,s:
            initital_result = (original_phi-single_mu)/single_s
            temp_phi = 1/(1+np.exp((-1.0)*initital_result))
            if count == 0:
                phi = temp_phi
            if count > 0:
                phi = np.concatenate((phi,temp_phi), axis=1) # join matrix
            count += 1
    else:
        assert(False), 'Unknown basis %s' % basis

    if with_bias:#Add bias column to Phi
        bias_column = np.ones((input.shape[0],1),dtype=np.int32)
        phi = np.hstack((bias_column,phi))
    return phi


def evaluate_regression(x_test,t_test,w,basis,degree=0,with_bias=True,reg_lambda=0, mu=[0], s=[1]):
    """Evaluate linear regression on a dataset.
    Args:
    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi = design_matrix(x_test,degree,basis,with_bias,mu,s)
    t_est = np.dot(phi,w)
    sum_error = 0.0
    for i in range(0,t_test.shape[0]):
        single_error = t_test[i]-t_est[i]
        single_error_squared = single_error **2
        sum_error = sum_error+single_error_squared
    # err =RMS
    err = 0.5*sum_error
    err = np.sqrt(2.0*err/t_test.shape[0])
    return (t_est, err)
