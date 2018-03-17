import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if X.ndim == 2:
        return softmax_2D(x)
    else:
        return softmax_1D(x)

def softmax_1D(X):
    c = np.max(X)
    exp_a = np.exp(X - c) # avoid overflow
    sum_exp_a = np.sum(exp_a)
    Y = exp_a / sum_exp_a

    return Y

def softmax_2D(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -1 * np.sum(t * np.log(y + delta)) / batch_size