import pandas as pd
import os
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from neuralnetwork import NeuralNetwork

bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, random_state=0)

y_trainv = y_train[:, np.newaxis]
y_testv = y_test[:, np.newaxis]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

print("sklearn logistic regression score: ", logreg.score(X_test_scaled, y_test))

# Needs a higher N to converge, but infeasible on my machine
Niter = 100000

# Higher eta leads to better convergence in the gradient, but encounters division by zero errors when calculating the cost function
eta = 0.01  
beta = np.random.randn(X_train.shape[1], 1)
#beta = np.zeros((X_train.shape[1], 1))

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

print("Iterate...")
for iter in range(Niter):
    sigmoid_list = sigmoid(np.dot(X_train,beta))
    negative_sigmoid_list = sigmoid(-np.dot(X_train,beta))
    
    gradient = X_train.T @ (sigmoid_list - y_trainv)
    beta -= eta*gradient
    #cost = -sum(y_trainv.T.dot(np.log(sigmoid_list)) + (1-y_trainv.T).dot(np.log(negative_sigmoid_list)))
    #print(cost)

    if iter%(Niter / 10) == 0:
        cost = -sum(y_trainv.T.dot(np.log(sigmoid_list)) + (1-y_trainv.T).dot(np.log(negative_sigmoid_list)))
        print("Iteration: ", iter, " Cost: ", cost[0])
#print(gradient)
print("Gradient norm: ", np.sqrt(sum(gradient**2))/gradient.size)
#print(beta)


y_fit = sigmoid(np.dot(X_test, beta))
y_fit = [1 if y_>0.5 else 0 for y_ in y_fit]
#print(y_fit)
#print(y_test)
indicator = 0
for fit, test in zip(y_fit, y_test):
    if fit == test:
        indicator += 1
accuracy = indicator/len(y_fit)
print("Accuracy score of implementation of logistic regression by gradient descent: ",accuracy)


# Neural network

runs = 50
my_acc = 0
skl_acc = 0

print("averaging over ", runs, " neural networks...")

for i in range(runs):
    testNet = NeuralNetwork(X_train_scaled, y_trainv, n_hidden_neurons=[50], n_categories=1, epochs=100, batch_size=100, eta=0.1, lmbd=0.01)
    testNet.train()
    y_fit = testNet.predict(X_test_scaled)
    indicator = 0
    for fit, test in zip(y_fit, y_test):
        if fit == test:
            indicator += 1
    accuracy = indicator/len(y_fit)
    from sklearn.neural_network import MLPClassifier
    dnn = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=0.01, learning_rate_init=0.1, max_iter=100)
    #dnn = MLPClassifier(activation='logistic', max_iter=100)
    dnn.fit(X_train_scaled, y_train)
    my_acc += accuracy / runs
    skl_acc += dnn.score(X_test, y_test) / runs
print("Accuracy score of cancer data prediction from implementation of nn: ", my_acc)
print("Accuracy score using sklearn MLPClassifier: ", skl_acc)
