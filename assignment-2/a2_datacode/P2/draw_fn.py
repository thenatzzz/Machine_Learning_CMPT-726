### PROBLEM 2 ###

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def problem1(x1, x2):
    return 6+2*(x1**2) + 2*(x2**2)
def problem2(x1,x2):
    return 8
def problem3_max_graph(prob1,prob2):
    return [max(l1, l2) for l1, l2 in zip(prob1, prob2)]
def draw():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = x2 = np.arange(-6.0, 6.0, 0.05)
    # x1 = x2 = np.arange(-3.0, 3.0, 0.05)
    X1, X2 = np.meshgrid(x1, x2)

    ##### PROBLEM 1
    zs1 = np.array([problem1(x1,x2) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z1 = zs1.reshape(X1.shape)
    ax.plot_surface(X1, X2, Z1) # uncomment to plot

    ##### PROBLEM 2
    zs2 = np.array([problem2(x1,x2) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z2 = zs2.reshape(X1.shape)
    # ax.plot_surface(X1, X2, Z2) # uncomment to plot

    ##### PROBLEM 3
    zs3 = problem3_max_graph(zs1,zs2)
    zs3 = np.asarray(zs3)
    Z3 = zs3.reshape(X1.shape)
    # ax.plot_surface(X1, X2, Z3) # uncomment to plot

    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5,5)
    ax.set_zlim3d(0,10)
    plt.savefig('problem.png')
    plt.show()

if __name__ == '__main__':
    draw()
