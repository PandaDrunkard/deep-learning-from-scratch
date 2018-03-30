from layer import Convolution, Pooling, Affine
import numpy as np

# Conv1: (99, 1, 28, 28) => (99, 30, 24, 24)
# Relu1: (99, 30, 24, 24) => (99, 30, 24, 24)
# Pool1: (99, 30, 24, 24) => (99, 30, 12, 12)
# Affine1: (99, 30, 12, 12) => (99, 100)
# Relu2: (99, 100) => (99, 100)
# Affine2: (99, 100) => (99, 10)
# W1: (30, 1, 5, 5)
# b1; (30,)
# W2: (4320, 100)
# b2: (100,)
# W3: (100, 10)
# b3: (10,)

def convolution_test():
    x = np.random.randn(99, 1, 28, 28)
    W = np.random.randn(30, 1, 5, 5)
    b = np.zeros(30)

    layer = Convolution(W, b, stride=1, pad=0)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)

def pooling_test():
    x = np.random.randn(99,30,24,24)
    
    layer = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)

def affine_test():
    x = np.random.randn(99,30,12,12)
    W = np.random.randn(30*12*12, 100)
    b = np.random.randn(100)

    layer = Affine(W, b)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)

convolution_test()
pooling_test()
affine_test()