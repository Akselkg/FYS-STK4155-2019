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
eps = 0.1
n = 20
file_id = "_order" + str(order) + "_eps" + str(eps) + "_n" + str(n)
print("OLS regression on franke_function with polynomials of order %d, epsilon = %f, n=%d(squared)" % (order, eps, n))

x = np.random.uniform(size=n)
y = np.random.uniform(size=n)
err = eps * np.random.normal(size=(n,n))

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
#trisurf_franke(x_flat, y_flat, z_tilde, filename="triangulate"+file_id)

# Plot regularized versions
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = franke_function(x,y)
X_plot = design_matrix(x.flatten(), y.flatten(), exponents)
z_tilde_plot = (X_plot @ beta).reshape(20,20)

#franke_plot(x, y, z, filename="franke_function"+file_id)
#franke_plot(x, y, z_tilde_plot, filename="franke_regression"+file_id)

#b)
print("--k-cross--")
order = 5
mse, r2, bias, variance = k_cross_Franke(n=20,eps=0.1, max_order=order)
print("R2 score: ", r2[order-1])
print("MSE: ", mse[order-1])
plt.figure()
plt.plot(range(1, order+1), bias, label="bias")
plt.plot(range(1, order+1), variance, label="variance")
plt.plot(range(1, order+1), mse, label="mse")
plt.xlabel("order")
plt.legend()
save_fig("k-cross validation")

#d)
print("--k-cross with ridge--")
mse, r2, bias, variance, l = k_cross_Franke_ridge(n=20, eps=0.1)
idx = np.argmin(mse)
print("R2 score: ", r2[idx])
print("MSE: ", mse[idx])

plt.figure()
plt.semilogx(l, bias, label="bias")
plt.semilogx(l, variance, label="variance")
plt.semilogx(l, mse, label="mse")
plt.xlabel("lambda")
plt.legend()
save_fig("k-cross with ridge")

#e)
print("--k-cross with lasso--")
mse, r2, bias, variance = franke_lasso(n=20, eps=0.1)

print("R2 score: ", r2)
print("MSE: ", mse)
print("Bias^2: ", bias)
print("Variance: ", variance)

#f)
# Load the terrain
terrain1 = imread(data_path('SRTM_data_Norway_1.tif'))
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('SRTM_data_Norway_1')

#g)

order = 5

min, max = np.min(terrain1), np.max(terrain1)
z = (terrain1 - min)/(max-min) # normalize terrain to values in [0,1]
x_len, y_len = np.shape(z)
x = np.array(range(x_len))
y = np.array(range(y_len))
x_mesh, y_mesh = np.meshgrid(x, y)
x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()
z_flat = z.flatten()

exponents = poly_exponents(order)
X = design_matrix(x_flat, y_flat, exponents)

H = np.linalg.inv(X.T.dot(X))
beta = H.dot(X.T).dot(z_flat)
z_tilde = X @ beta

print("R2 score: %f" % R2(z_flat, z_tilde))
print("MSE: %f" % MSE(z_flat, z_tilde))