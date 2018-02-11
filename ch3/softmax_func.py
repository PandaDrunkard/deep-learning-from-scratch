import numpy as np

def softmax(X):
    exp_a = np.exp(X)
    sum_exp_a = np.sum(exp_a)
    Y = exp_a / sum_exp_a

    return Y