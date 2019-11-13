import numpy as np

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

class NeuralNetwork:
    """
    Built on top of the NeuralNetwork class from lecture notes to support multiple layers.
    To support this the input variable n_hidden_neurons should be a list of the number of hidden neurons in each hidden layer,
    starting from the first hidden layer.
    """
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons = [50],
            n_categories = 10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_neurons = [self.n_features] + n_hidden_neurons + [n_categories]
        self.n_hidden_layers = len(self.n_neurons) - 2

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()
    

    def create_biases_and_weights(self):
        self.weights = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.bias = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)

        for i in range(self.n_hidden_layers+1):
            self.weights[i] = np.random.randn(self.n_neurons[i], self.n_neurons[i+1])
            self.bias[i] = np.zeros(self.n_neurons[i+1]) + 0.01

    def feed_forward(self):
        self.a_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.a_array[0] = self.X_data

        for i in range(self.n_hidden_layers + 1):
            z = self.a_array[i] @ self.weights[i] + self.bias[i]
            self.a_array[i+1] = sigmoid(z)
        
        #exp_term = np.exp(z)
        #self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        self.probabilities = self.a_array[i+1]
        

    def feed_forward_out(self, X):
        a_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        a_array[0] = X

        for i in range(self.n_hidden_layers + 1):
            z = a_array[i] @ self.weights[i] + self.bias[i]
            a_array[i+1] = sigmoid(z)
        
        #exp_term = np.exp(z)
        #probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        probabilities = a_array[i+1]
        return probabilities

    def backpropagation(self):
        error_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.weights_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.bias_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)   

        error_array[-1] = self.probabilities - self.Y_data
        for i in range(self.n_hidden_layers + 1, 0, -1):
            error_array[i-1] = (error_array[i] @ self.weights[i-1].T) * self.a_array[i-1] * (1 - self.a_array[i-1])
            self.weights_gradient_array[i-1] = self.a_array[i-1].T @ error_array[i]
            self.bias_gradient_array[i-1] = np.sum(error_array[i], axis=0)

            if self.lmbd > 0.0:
                self.weights_gradient_array[i-1] += self.lmbd * self.weights_gradient_array[i-1]

            self.weights[i-1] -= self.eta * self.weights_gradient_array[i-1]
            self.bias[i-1] -= self.eta * self.bias_gradient_array[i-1]


    def predict(self, X):
        
        probabilities = self.feed_forward_out(X)
        # If binary category, "yes/no"
        if self.n_neurons[-1] == 1:
            return np.rint(probabilities)  # round sigmoid to nearest int
        else:
            return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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

    runs = 100
    my_acc = 0
    skl_acc = 0
    for i in range(runs):
        testNet = NeuralNetwork(X_train_scaled, y_trainv, n_hidden_neurons=[50], n_categories=1, epochs=1000, batch_size=100, eta=0.1, lmbd=0.01)
        testNet.train()
        y_fit = testNet.predict(X_test_scaled)
        indicator = 0

        for fit, test in zip(y_fit, y_test):
            if fit == test:
                indicator += 1
        accuracy = indicator/len(y_fit)


        from sklearn.neural_network import MLPClassifier
        #dnn = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=0.01, learning_rate_init=0.1, max_iter=100)
        dnn = MLPClassifier(activation='logistic', max_iter=1000)
        dnn.fit(X_train_scaled, y_train)

        my_acc += accuracy / runs
        skl_acc += dnn.score(X_test, y_test) / runs

    print("Accuracy score of cancer data prediction from implementation of nn: ", my_acc)
    print("Accuracy score using sklearn MLPClassifier: ", skl_acc)

    testNet = NeuralNetwork(X_train_scaled, y_trainv, n_hidden_neurons=[50], n_categories=1, epochs=100, batch_size=100, eta=0.1, lmbd=0.01)
    testNet.train()
    y_fit = testNet.predict(X_test_scaled)
    indicator = 0

    for fit, test in zip(y_fit, y_test):
        if fit == test:
            indicator += 1
    accuracy = indicator/len(y_fit)
    #print("Accuracy score of cancer data prediction from implementation of nn: ", accuracy)

    from sklearn.neural_network import MLPClassifier
    dnn = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=0.01, learning_rate_init=0.1, max_iter=100)
    #dnn = MLPClassifier(activation='logistic', max_iter=2000)
    dnn.fit(X_train_scaled, y_train)

    #print("Accuracy score using sklearn MLPClassifier: ", dnn.score(X_test, y_test))
