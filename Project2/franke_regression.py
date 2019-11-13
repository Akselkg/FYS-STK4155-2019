import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from pro1_functions import *
from neuralnetwork import NeuralNetwork

from imageio import imread
import os

np.random.seed(2019)

# Design matrix
order = 5
eps_val = 0.1
n = 40
file_id = "_order" + str(order) + "_eps" + str(eps_val) + "_n" + str(n)
print("OLS regression on franke_function with polynomials of order %d, epsilon = %f, n=%d(squared)" % (order, eps_val, n))

x = np.random.uniform(size=n)
y = np.random.uniform(size=n)
err = eps_val * np.random.normal(size=(n,n))

x_mesh, y_mesh = np.meshgrid(x, y)
x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()

z_mesh = franke_function(x_mesh, y_mesh) + err
z_flat = z_mesh.flatten()

exponents = poly_exponents(order)
X = design_matrix(x_flat, y_flat, exponents)

# Regression

H = np.linalg.inv(X.T.dot(X))
beta = H.dot(X.T).dot(z_flat)
z_tilde = X @ beta

print("R2 score: %f" % R2(z_flat, z_tilde))
print("MSE: %f" % MSE(z_flat, z_tilde))

# Neural network

order = 5
eps_val = 0.1
n = 40
file_id = "_order" + str(order) + "_eps" + str(eps_val) + "_n" + str(n)
print("Neural network fit on franke_function with polynomials of order %d, epsilon = %f, n=%d(squared)" % (order, eps_val, n))

x = np.random.uniform(size=n)
y = np.random.uniform(size=n)
err = eps_val * np.random.normal(size=(n,n))

x_mesh, y_mesh = np.meshgrid(x, y)
x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()

z_mesh = franke_function(x_mesh, y_mesh) + err
z_flat = z_mesh.flatten()

exponents = poly_exponents(order)
X = design_matrix(x_flat, y_flat, exponents)

X_train, X_test, y_train, y_test = train_test_split(X, z_flat, random_state=0)
y_trainv = y_train[:, np.newaxis]
y_testv = y_test[:, np.newaxis]


my_mse = 0
my_r2 = 0

print("training neural network...")

testNet = NeuralNetwork(X_train, y_trainv, problem_type='regression', n_hidden_neurons=[100], n_categories=1, epochs=1000, batch_size=100, eta=0.001, lmbd=0.0001)
testNet.train()
y_fit = testNet.predict_values(X_test)
mse = MSE(y_test, y_fit.flatten())
r2 = R2(y_test, y_fit.flatten())

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = franke_function(x,y)
X_plot = design_matrix(x.flatten(), y.flatten(), exponents)
z_tilde_plot = (X_plot @ beta).reshape(20,20)
z_tilde_nn = testNet.predict_values(X_plot)

franke_plot(x, y, z, filename="franke_function"+file_id)
franke_plot(x, y, z_tilde_plot, filename="franke_regression"+file_id)
franke_plot(x, y, z_tilde_nn.reshape(20,20), filename="franke_neural"+file_id)

print("test data:")
print("R2 score: %f" % r2)
print("MSE: %f" % mse)
print("plot data:")
print("R2 score: %f" % R2(z.flatten(), z_tilde_nn.flatten()))
print("MSE: %f" % MSE(z.flatten(), z_tilde_nn.flatten()))
