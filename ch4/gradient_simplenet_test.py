import numpy as np
import sys, os
sys.path.append(os.pardir)
from gradient_simplenet import simpleNet
from common.gradient import numerical_gradient

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

p = net.predict(x)
l = net.loss(x, t)
print(p)
print(np.argmax(p))
print(l)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
