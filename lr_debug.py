from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
os.chdir('D:/Krishna/Acads/Q4/ML/HW')
try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    trys=int(1e5)
    i = 0
    theta_frame=np.zeros((trys,3))
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if i%100 == 0:   
            theta_frame[int(i/100),:]=theta
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
        if i>=trys-1:
            break
    theta_frame=theta_frame[:int(i/100)+1]
    return theta_frame

def cost_fun(Xb,Yb,theta):
    m, n = Xb.shape
    Ypred=1/(1+np.exp(-Xb.dot(theta)))
    J=1/2/m*np.sum((Yb-Ypred)**2)
    return J


#
#
#print('==== Training model on data set A ====')
#Xa, Ya = load_data('data_a.txt')
#theta_frame=logistic_regression(Xa, Ya)
#cost=[cost_fun(Xa,Ya,theta) for theta in theta_frame]

print('\n==== Training model on data set B ====')
Xb, Yb = load_data('data_b.txt')
theta_frame=logistic_regression(Xb, Yb)
#cost=[cost_fun(Xb,Yb,theta) for theta in theta_frame]



plt.style.use('seaborn-darkgrid')

theta1,theta2 = np.meshgrid(theta_frame[:,0],theta_frame[:,1])
Z=np.copy(theta1)
for i in range(theta1.shape[0]):
    for j in range(theta1.shape[1]):
        theta=[theta1[i,j],theta2[i,j]]
        Z[i,j]=cost_fun(Xb[:,:2], Yb,theta)        

#fig, ax = plt.subplots(1,1,figsize=(4,4))

fig=plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta1,theta2,Z,cmap=cm.coolwarm)
#plt.xscale('symlog')
#plt.yscale('symlog')