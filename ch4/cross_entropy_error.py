import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -1 * np.sum(t * np.log(y + delta))

def test_cross_entropy_error():
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

    d = cross_entropy_error(y, t)
    print(d)

    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    d = cross_entropy_error(y, t)
    print(d)

# test_cross_entropy_error()