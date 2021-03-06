import sys, os
sys.path.append(os.pardir)
from common.layer import *
import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])

relu_layer = Relu()
relu_out = relu_layer.forward(x)
relu_dx = relu_layer.backward(relu_out)

print(x)
print(relu_out)
print(relu_dx)

sigmoid_layer = Sigmoid()
s_out = sigmoid_layer.forward(x)
s_dx = sigmoid_layer.backward(s_out)

print(s_out)
print(s_dx)

W = np.random.randn(2,10)
b = np.random.randn(2,1)

affine_layer = Affine(W, b)
affine_fw = affine_layer.forward(x)
affine_bw = affine_layer.backward(affine_fw)

print(affine_fw)
print(affine_bw)

