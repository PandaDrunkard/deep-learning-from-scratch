import numpy as np

def softmax(X):
    c = np.max(X)
    exp_a = np.exp(X - c) # avoid overflow
    sum_exp_a = np.sum(exp_a)
    Y = exp_a / sum_exp_a

    return Y