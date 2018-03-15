import numpy as np

def softmax(X):
    c = np.max(X)
    exp_a = np.exp(X - c) # avoid overflow
    sum_exp_a = np.sum(exp_a)
    Y = exp_a / sum_exp_a

    return Y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -1 * np.sum(t * np.log(y + delta)) / batch_size