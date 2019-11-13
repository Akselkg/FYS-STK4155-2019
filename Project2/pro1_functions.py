import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os


def image_path(figure_id):
    PROJECT_ROOT_DIR = "Results"
    if not os.path.exists(PROJECT_ROOT_DIR):
        os.mkdir(PROJECT_ROOT_DIR)
    return os.path.join(PROJECT_ROOT_DIR, figure_id)

def data_path(data_id):
    DATA_DIR = "Datafiles/"
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    return os.path.join(DATA_DIR, data_id)

def save_fig(figure_id):
    plt.savefig(image_path(figure_id) + ".png", format='png')

def franke_plot(x, y, z, colormap=cm.coolwarm, filename="franke_function"):   
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=colormap, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    save_fig(filename)

# plot non-uniform data as a surface by triangulation
def trisurf_franke(x_flat, y_flat, z, colormap=cm.coolwarm, filename="triangulate"):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    surf = ax.plot_trisurf(x_flat, y_flat, z, cmap=colormap)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    save_fig(filename)

# creates list of all tuples (i,j) so that (x^i*y^j) is of order at most n
def poly_exponents(n):
    exponents = []
    for i in range(n+1):
        for j in range(i+1):
            exponents.append((j, i-j))
    return exponents

def franke_function(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# creates matrix [1, x, y, x^2, xy, y^2, ...] x, y, column vectors
def design_matrix(x_flat, y_flat, exponents):
    X = np.zeros((len(x_flat), len(exponents)))
    for i, exponent_pair in enumerate(exponents):
        X[:,i] = (x_flat ** exponent_pair[0]) * (y_flat ** exponent_pair[1])
    return X

def MSE(y, y_tilde):
    return np.mean((y - y_tilde)**2)

def R2(y, y_tilde):
    return 1 - (np.mean((y - y_tilde)**2) / np.mean((y - np.mean(y))**2))

def calc_bias(y, y_tilde):
    return (np.mean(y) - np.mean(y_tilde))**2

def calc_variance(y_tilde):
    return np.mean((y_tilde - np.mean(y_tilde))**2)


def k_cross_Franke(n=10, k=10, eps=0, max_order=5):
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    err = eps * np.random.normal(size=(n*n))
    x_mesh, y_mesh = np.meshgrid(x, y)
    x, y = x_mesh.flatten(), y_mesh.flatten()
    z_mesh = franke_function(x_mesh, y_mesh)
    z = z_mesh.flatten() + err

    return k_cross(x, y, z, k, max_order)

def k_cross(x, y, z, k=10, max_order=5):

    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k); sub_arrays_z = np.array_split(z, k)
    test_errors = np.zeros((k, max_order))
    R2_scores = np.zeros((k, max_order))
    biases = np.zeros((k, max_order))
    variances = np.zeros((k, max_order))

    train_errors = np.zeros((k, max_order))



    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        y_valid = sub_arrays_y[k_]
        z_train = np.concatenate(sub_arrays_z[:k_] + sub_arrays_z[k_+1:])
        z_valid = sub_arrays_z[k_]

        for order in range(1, max_order+1):
            exponents = poly_exponents(order)
            xb = design_matrix(x_train, y_train, exponents)
            X_val = design_matrix(x_valid, y_valid, exponents)
            beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z_train)
            zhat = X_val @ beta
            test_errors[k_, order-1] = MSE(z_valid, zhat)
            R2_scores[k_, order-1] = R2(z_valid, zhat)
            biases[k_, order-1] = calc_bias(z_valid, zhat)
            variances[k_, order-1] = calc_variance(zhat)

            z_train_hat = xb @ beta
            train_errors[k_, order-1] = MSE(z_train, z_train_hat)

    test_error = np.mean(test_errors, axis=0)
    r2 = np.mean(R2_scores, axis=0)
    bias = np.mean(biases, axis=0)
    variance = np.mean(variances, axis=0)

    train_error = np.mean(train_errors, axis=0)

    return test_error, r2, bias, variance, train_error

def k_cross_Franke_ridge(n=10, k=10, eps=0, order=5):

    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    err = eps * np.random.normal(size=(n*n))
    x_mesh, y_mesh = np.meshgrid(x, y)
    x, y = x_mesh.flatten(), y_mesh.flatten()
    z_mesh = franke_function(x_mesh, y_mesh)
    z = z_mesh.flatten() + err

    return k_cross_ridge(x, y, z, k, order)

def k_cross_ridge(x, y, z, k=10, order=5):
    m = 50
    l = np.exp(np.linspace(-20, 3, m))
    
    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k); sub_arrays_z = np.array_split(z, k)
    test_errors = np.zeros((k, m))
    R2_scores = np.zeros((k, m))
    biases = np.zeros((k, m))
    variances = np.zeros((k, m))

    train_errors = np.zeros((k, m))

    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        y_valid = sub_arrays_y[k_]
        z_train = np.concatenate(sub_arrays_z[:k_] + sub_arrays_z[k_+1:])
        z_valid = sub_arrays_z[k_]

        for i_l, l_ in enumerate(l):
            exponents = poly_exponents(order)
            xb = design_matrix(x_train, y_train, exponents)
            X_val = design_matrix(x_valid, y_valid, exponents)

            l_eye = l_ * np.eye(xb.shape[1])
            beta = np.linalg.inv(xb.T.dot(xb) + l_eye).dot(xb.T).dot(z_train)
            zhat = X_val @ beta
            test_errors[k_, i_l] = MSE(z_valid, zhat)
            R2_scores[k_, i_l] = R2(z_valid, zhat)
            biases[k_, i_l] = calc_bias(z_valid, zhat)
            variances[k_, i_l] = calc_variance(zhat)

            z_train_hat = xb @ beta
            train_errors[k_, i_l] = MSE(z_train, z_train_hat)

    test_error = np.mean(test_errors, axis=0)
    r2 = np.mean(R2_scores, axis=0)
    bias = np.mean(biases, axis=0)
    variance = np.mean(variances, axis=0)

    train_error = np.mean(train_errors, axis=0)

    return test_error, r2, bias, variance, train_error, l

def franke_lasso(n=10, eps=0.1):
    max_order = 5
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    err = eps * np.random.normal(size=(n*n))
    
    x_mesh, y_mesh = np.meshgrid(x, y)
    x, y = x_mesh.flatten(), y_mesh.flatten()

    x_train = x[:int(n*n/2)]; y_train = y[:int(n*n/2)]; err_train = err[:int(n*n/2)]
    x_valid = x[int(n*n/2):]; y_valid = y[int(n*n/2):]; err_valid = err[int(n*n/2):]

    z_train = franke_function(x_train, y_train) + err_train
    z_valid = franke_function(x_valid, y_valid) + err_valid
    exponents = poly_exponents(max_order)
    xb = design_matrix(x_train, y_train, exponents)[:,1:] # cut intercept column

    lasso=linear_model.LassoCV(max_iter=100000, cv=10)
    lasso.fit(xb, z_train)
    predl=lasso.predict(design_matrix(x_valid, y_valid, exponents)[:,1:])

    return MSE(z_valid, predl), R2(z_valid, predl), calc_bias(z_valid, predl), calc_variance(predl)

## old bad k_cross
"""
def k_cross_Franke(n=10, k=10, eps=0, max_order=5):
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    err = eps * np.random.normal(size=(n*n))
    x_mesh, y_mesh = np.meshgrid(x, y)
    x, y = x_mesh.flatten(), y_mesh.flatten()
    z_mesh = franke_function(x_mesh, y_mesh)
    z = z_mesh.flatten()
    
    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k); sub_arrays_err = np.array_split(err, k)
    mses = np.zeros((k, max_order))
    R2_scores = np.zeros((k, max_order))
    biases = np.zeros((k, max_order))
    variances = np.zeros((k, max_order))

    beta_list = []
    betas = []
    mse_list = []
    R2_score_list = []
    bias_list = []
    variance_list =[]


    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        y_valid = sub_arrays_y[k_]
        err_train = np.concatenate(sub_arrays_err[:k_] + sub_arrays_err[k_+1:])
        err_valid = sub_arrays_err[k_]

        z_train = franke_function(x_train, y_train) + err_train
        z_valid = franke_function(x_valid, y_valid) + err_valid

        beta_list.append([])

        for order in range(1, max_order+1):
            exponents = poly_exponents(order)
            xb = design_matrix(x_train, y_train, exponents)
            X_val = design_matrix(x_valid, y_valid, exponents)
            beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z_train)
            zhat = X_val @ beta
            mses[k_, order-1] = MSE(z_valid, zhat)
            R2_scores[k_, order-1] = R2(z_valid, zhat)
            biases[k_, order-1] = calc_bias(z_valid, zhat)
            variances[k_, order-1] = calc_variance(zhat)

            beta_list[k_].append(beta)

    # find mean betas
    for orderidx in range(max_order):
        betasum = np.zeros(len(poly_exponents(orderidx+1)))
        for k_ in range(k):
            
            betasum += np.array(beta_list[k_][orderidx])
        betas.append(betasum/k)

    #test fit on mean betas
    for order in range(1, max_order+1):
        exponents = poly_exponents(order)
        X_val = design_matrix(x, y, exponents)
        zhat = X_val @ betas[order-1]
        mse_list.append(MSE(z, zhat))
        R2_score_list.append(R2(z, zhat))
        bias_list.append(calc_bias(z, zhat))
        variance_list.append(calc_variance(zhat))
    
    return mse_list, R2_score_list, bias_list, variance_list
"""