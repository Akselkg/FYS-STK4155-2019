import autograd.numpy as np
from autograd import jacobian, hessian, grad

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1-sigmoid(z))

def reLU(z):
    return np.where(z < 0, 0, z)

def reLU_deriv(z):
    return np.where(z < 0, 0, 1)


class NeuralNetwork:
    """
    Built on top of the NeuralNetwork class from lecture notes to support:
        Multiple layers,
        Both classification(binary/multiple) and regression
        Custom activation functions(but you need to manually edit the initialize_functions(self) method as of now)
    The input variable n_hidden_neurons should be a list of the number of hidden neurons in each hidden layer,
    starting from the first hidden layer.
    n_categories=1 for regression problems
    """
    def __init__(
            self,
            X_data,
            Y_data,
            problem_type = 'classification',
            n_hidden_neurons = [50],
            n_categories = 10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.problem_type = problem_type

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
        self.initialize_functions()
    
    def initialize_functions(self):
        """
        Generate activation functions and corresponding derivatives for each layer in the network.
        """
        self.actfunc_list = []
        self.actderiv_list = []
        
        if self.problem_type=='classification' or self.problem_type=='pde' :
            for i in range(self.n_hidden_layers+1): 
                self.actfunc_list.append(sigmoid)
                self.actderiv_list.append(sigmoid_deriv)
        elif self.problem_type=='regression':
            for i in range(self.n_hidden_layers+1): 
                self.actfunc_list.append(reLU)
                self.actderiv_list.append(reLU_deriv)

    def create_biases_and_weights(self):
        self.weights = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.bias = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)

        for i in range(self.n_hidden_layers+1):
            self.weights[i] = np.random.randn(self.n_neurons[i], self.n_neurons[i+1])
            self.bias[i] = np.zeros(self.n_neurons[i+1]) + 0.01

    def feed_forward(self):
        self.a_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.a_deriv_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.a_array[0] = self.X_data

        for i in range(self.n_hidden_layers + 1):
            z = self.a_array[i] @ self.weights[i] + self.bias[i]
            self.a_array[i+1] = self.actfunc_list[i](z)
            self.a_deriv_array[i+1] = self.actderiv_list[i](z)
        
        if self.n_neurons[-1] > 1 and self.problem_type == 'classification':
            exp_term = np.exp(z)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)          
        else:
            self.probabilities = self.a_array[i+1]          
        

    def feed_forward_out(self, X):
        a_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        a_array[0] = X

        for i in range(self.n_hidden_layers + 1):
            z = a_array[i] @ self.weights[i] + self.bias[i]
            a_array[i+1] = self.actfunc_list[i](z)
        
        if self.n_neurons[-1] > 1 and self.problem_type == 'classification':
            exp_term = np.exp(z)
            probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)          
        else:
            probabilities = a_array[i+1]   
        
        return probabilities

    def backpropagation(self):
        error_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.weights_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.bias_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)   

        error_array[-1] = self.probabilities - self.Y_data
        if self.problem_type == 'regression':
            error_array[-1] * self.a_deriv_array[-1]
        
        for i in range(self.n_hidden_layers + 1, 0, -1):
            error_array[i-1] = (error_array[i] @ self.weights[i-1].T) * self.a_deriv_array[i-1]
            self.weights_gradient_array[i-1] = self.a_array[i-1].T @ error_array[i]
            self.bias_gradient_array[i-1] = np.sum(error_array[i], axis=0)

            if self.lmbd > 0.0:
                self.weights_gradient_array[i-1] += self.lmbd * self.weights_gradient_array[i-1]

            self.weights[i-1] -= self.eta * self.weights_gradient_array[i-1]
            self.bias[i-1] -= self.eta * self.bias_gradient_array[i-1]
    
    def u(self, x):
        return np.sin(np.pi*x)

    def g_trial(self, x, t):
        return (1-t)*u(x) + x*(1-x)*t*pde_predict(x, t)

    def pde_predict(self, x, t):
        
        x_mesh, t_mesh = np.meshgrid(x, t)
        x_flat, t_flat = x_mesh.flatten(), t_mesh.flatten()

    
    def pde_backprop(self):
        error_array = np.zeros(self.n_hidden_layers+2, dtype=np.ndarray)
        self.weights_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.bias_gradient_array = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)   

        error_array[-1] = self.probabilities - self.Y_data
        if self.problem_type == 'regression':
            error_array[-1] * self.a_deriv_array[-1]
        
        for i in range(self.n_hidden_layers + 1, 0, -1):
            error_array[i-1] = (error_array[i] @ self.weights[i-1].T) * self.a_deriv_array[i-1]
            self.weights_gradient_array[i-1] = self.a_array[i-1].T @ error_array[i]
            self.bias_gradient_array[i-1] = np.sum(error_array[i], axis=0)

            if self.lmbd > 0.0:
                self.weights_gradient_array[i-1] += self.lmbd * self.weights_gradient_array[i-1]

            self.weights[i-1] -= self.eta * self.weights_gradient_array[i-1]
            self.bias[i-1] -= self.eta * self.bias_gradient_array[i-1]


    # For classisfication
    def predict(self, X):
        
        probabilities = self.feed_forward_out(X)
        # If binary category, "yes/no"
        if self.n_neurons[-1] == 1:
            return np.rint(probabilities)  # round sigmoid to nearest int(0/1)
        else:
            return np.argmax(probabilities, axis=1)

    # For classification
    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities
    
    # For regression
    def predict_values(self, X):
        values = self.feed_forward_out(X)
        return values

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
    pass
