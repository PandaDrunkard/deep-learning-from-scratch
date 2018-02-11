import numpy as np
from sigmoid_func import sigmoid

def init_network():
    network = {}
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['B1'] = np.array([
        0.1, 
        0.2, 
        0.3
    ])
    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    network['B2'] = np.array([
        0.1,
        0.2
    ])
    network['W3'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    network['B3'] = np.array([
        0.1,
        0.2
    ])

    return network

def forward(network, X):
    print(network)

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)

    print(A1)
    print(Z1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    print(A2)
    print(Z2)

    A3 = np.dot(Z2, W3) + B3
    Y = identity_func(A3)

    print(A3)

    return Y

def identity_func(X):
    return X


network = init_network()

X = np.array([1.0, 0.5])

Y = forward(network, X)

print(Y)