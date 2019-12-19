import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf     ## if using tensorflow 1.x

import numpy as np
from pro1_functions import MSE
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# initialize
tf.set_random_seed(1729)
np.random.seed(1729)

N = 5
x0 = 2 * np.random.rand(N) - 1
Q = 2 * np.random.rand(N, N) - 1
A = (Q.T + Q) / 2
ones = np.ones(5)
I = np.diag(ones)

print("True eigenvectors as calculated by numpy.linalg.eig(A):")
print(np.linalg.eig(A))

Nt = 10
t = np.linspace(0,1, Nt)
t_test = np.stack([t for i in range(N)])


## The construction phase

t_tf = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))
t_test_tf = tf.einsum('ij->ji', tf.convert_to_tensor(t_test))
ones_tf = tf.convert_to_tensor(ones)
x0 = tf.convert_to_tensor(x0)
zeros = tf.convert_to_tensor(np.zeros((Nt, N)))
A = tf.convert_to_tensor(A)
I = tf.convert_to_tensor(I)

num_iter = 10000
num_hidden_neurons = [100, 50, 25]


with tf.variable_scope('dnn'):
    num_hidden_layers = np.size(num_hidden_neurons)

    previous_layer = t_tf

    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.sigmoid)
        previous_layer = current_layer

    dnn_output = tf.layers.dense(previous_layer, N)

def f(x):

    #f_out = np.zeros(x.shape)
    #print(f_out[0].shape)
    dot_x = tf.einsum('ti,ti->t',x,x)
    xAx_prod = 1 - tf.einsum('ti,ti->t',tf.einsum('ti,ij->tj', x, A), x)
    f_out = tf.einsum('tij, tj->ti', (tf.einsum('t,ij->tij',dot_x, A) + tf.einsum('t,ij->tij',xAx_prod, I)), x)
    #for i in range(Nt):
    #    t_sum = tf.einsum('ij, j->i', (tf.einsum('i,i->',x[i],x[i]) * A + (1 - tf.einsum('i,i->',tf.einsum('i,ij->j', x[i], A), x[i]))*I), x[i])
    #    print(t_sum.shape)
    #    f_out[i] += t_sum
    print(f_out.shape)
    return f_out


with tf.name_scope('loss'):
    g_trial = x0 + t_tf*dnn_output

    g_trial_dt = tf.einsum('ij,k->ik', tf.gradients(g_trial,t_tf)[0], ones_tf)
    g_trial = x0 + tf.einsum('ij,ij->ij', t_test_tf,dnn_output)
    g_trial_dt = tf.gradients(g_trial,t_test_tf)[0]
    #print(g_trial.shape)
    #print(g_trial_dt.shape)
    loss = tf.losses.mean_squared_error(zeros, g_trial_dt + g_trial - f(g_trial))

    #loss = tf.losses.mean_squared_error(zeros, - g_trial + f(g_trial))

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

g_dnn = None

## The execution phase
with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        sess.run(traning_op)

        # If one desires to see how the cost function behaves during training
        if i % 500 == 0:
            print(loss.eval())

    g_dnn = g_trial.eval()


## Compare with the analytical solution



print("Computed eigenvector through nn")
print(g_dnn[-1])