import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    # xと同じ形だが、要素が0.であるarrayを返す
    grad = np.zeros_like(x)

    for idx in range(x.size):
        act_x = x[idx]
        # f(x+h)
        x[idx] = act_x + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = act_x - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = act_x
    
    return grad