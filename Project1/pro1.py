import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from project_functions import *
from imageio import imread
import os

np.random.seed(2019)

# Design matrix
order = 5
eps_val = 0
n = 20
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

print("Regression parameter variance:")
var_est = (1.0/(n*n - order)) * np.mean((z_flat - z_tilde)**2)  ## order = p + 1
for j in range(order+1):
    beta_j_var = var_est * np.sqrt(H[j,j])
    print("Beta_" + str(j) + " = " + str(beta_j_var))
    

print("R2 score: %f" % R2(z_flat, z_tilde))
print("MSE: %f" % MSE(z_flat, z_tilde))

# --- plotting --- #

x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()
trisurf_franke(x_flat, y_flat, z_tilde, filename="triangulate"+file_id)

# Plot regularized versions
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = franke_function(x,y)
X_plot = design_matrix(x.flatten(), y.flatten(), exponents)
z_tilde_plot = (X_plot @ beta).reshape(20,20)

franke_plot(x, y, z, filename="franke_function"+file_id)
franke_plot(x, y, z_tilde_plot, filename="franke_regression"+file_id)

#b)
print("--k-cross--")
order = 10
mse, r2, bias, variance, train_err = k_cross_Franke(n=20,eps=eps_val, max_order=order)
best_order = np.argmax(r2)
print("best order:", best_order)
print("R2 score: ", r2[best_order])
print("MSE: ", mse[best_order])

plt.figure()
plt.semilogy(range(1, order+1), mse, label="test_err")
plt.semilogy(range(1, order+1), train_err, label="train_err")
plt.xlabel("Order")
plt.title("10-cross validation on the Franke function for different order polynomials")
plt.legend()
save_fig("k-cross validation log")

#d)
print("--k-cross with ridge--")
mse, r2, bias, variance, train_err, l = k_cross_Franke_ridge(n=20, eps=eps_val)
idx = np.argmax(r2)
print("Best lambda: ", l[idx])
print("R2 score: ", r2[idx])
print("MSE: ", mse[idx])

## logplot
plt.figure()
plt.loglog(l, mse, label="test_err")
plt.loglog(l, train_err, label="train_err")
plt.xlabel("lambda")
plt.legend()
save_fig("k-cross with ridge log")

#e)
print("--k-cross with lasso--")
mse, r2, bias, variance = franke_lasso(n=20, eps=eps_val)

print("R2 score: ", r2)
print("MSE: ", mse)
print("Bias^2: ", bias)
print("Variance: ", variance)

#f)
# Load the terrain
terrain1 = imread(data_path('SRTM_data_Norway_1.tif'))
terrain1 = terrain1[-1000:, -1000:]

print("-Image data-")
# Show the terrain
plt.figure()
plt.title('Terrain data from Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1')

#g)

order = 7

min_val, max_val = np.min(terrain1), np.max(terrain1)
z = (terrain1[::4, ::4] - min_val)/(max_val-min_val) # normalize terrain to values in [0,1]
x_len, y_len = np.shape(z)
x = np.linspace(0, 1, num=x_len)
y = np.linspace(0, 1, num=y_len)
x_mesh, y_mesh = np.meshgrid(x, y)
x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()
z_flat = z.flatten()

exponents = poly_exponents(order)
X = design_matrix(x_flat, y_flat, exponents)

H = np.linalg.inv(X.T.dot(X))
beta = H.dot(X.T).dot(z_flat)
z_tilde = X @ beta

plt.figure()
plt.title('Terrain data from Norway after regression')

plt.imshow(np.reshape(z_tilde, np.shape(z)), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1_regression')

print("OLS:")
print("R2 score: %f" % R2(z_flat, z_tilde))
print("MSE: %f" % MSE(z_flat, z_tilde))

## k-cross OLS on terrain

print("--k-cross terrain data--")
order = 10
mse, r2, bias, variance, train_err = k_cross(x_flat, y_flat, z_flat, max_order=order)
best_order = np.argmax(r2)
print("best order:", best_order)
print("R2 score: ", r2[best_order])
print("MSE: ", mse[best_order])

plt.figure()

plt.semilogy(range(1, order+1), mse, label="test error")
plt.semilogy(range(1, order+1), train_err, label="training error")
plt.xlabel("Maximum order of regression coefficients")
plt.title("10-cross validation on terrain data for different order polynomials")
plt.legend()
save_fig("k-cross validation terrain")

# k-cross ridge regression on terrain

print("--k-cross ridge terrain data--")

mse, r2, bias, variance, train_err, l = k_cross_ridge(x_flat, y_flat, z_flat, order=7)

idx = np.argmax(r2)
print("Best lambda: ", l[idx])
print("R2 score: ", r2[idx])
print("MSE: ", mse[idx])

## logplot
plt.figure()
plt.loglog(l, mse, label="test error")
plt.loglog(l, train_err, label="training error")
plt.xlabel("lambda")
plt.legend()
plt.title("10-cross validation on terrain data for different order polynomials")
save_fig("k-cross with ridge terrain")

## generate terrain image with ridge regression

l_eye = l[idx] * np.eye(X.shape[1])
H_ridge = np.linalg.inv(X.T.dot(X) + l_eye)
beta_ridge = H_ridge.dot(X.T).dot(z_flat)
z_tilde_ridge = X @ beta_ridge

plt.figure()
plt.title('Terrain data from Norway after ridge regression')

plt.imshow(np.reshape(z_tilde_ridge, np.shape(z)), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1_ridge_regression')

# lasso

lasso=linear_model.LassoCV(max_iter=1000000, cv=5)
lasso.fit(X[:,1:], z_flat)
predl=lasso.predict(X[:,1:])

plt.figure()
plt.title('Terrain data from Norway after lasso')

plt.imshow(np.reshape(predl, np.shape(z)), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1_lasso')

## compressing image data (unused)
r = 20
U, S, V = np.linalg.svd(z)        #using SVD method to decompose image

Y  = S[:r]                # Cut off "small" singular values
Ynew = np.zeros(np.shape(z)) 

for i in range(r):
    Ynew = Ynew + Y[i]* np.outer(U[: , i], V.T[: , i])

plt.figure()
plt.title('Terrain data from Norway compressed')
plt.imshow(Ynew, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1_compressed')

