from sigmoid_func import sigmoid
import numpy as np
import matplotlib.pyplot as plt

def draw_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

draw_sigmoid()