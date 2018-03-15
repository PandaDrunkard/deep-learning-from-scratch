import numpy as np
from numerical_gradient import numerical_gradient
from function_2 import function_2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    ret_x = np.copy(init_x)

    for i in range(step_num):
        grad = numerical_gradient(f, ret_x)
        ret_x -= lr * grad
    
    return ret_x

def test_gd():
    init_x = np.array([-3.0, 4.0])
    lim_x = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
    print(init_x)
    print(lim_x)

# test_gd()