from sigmoid_func import sigmoid
from step_function import step_function
import numpy as np
import matplotlib.pyplot as plt

def compare_functions():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)
    plt.plot(x, y1, label="sigmoid")
    plt.plot(x, y2, label="step", linestyle="--")
    plt.ylim(-0.1, 1.1)
    plt.title("sigmoid & step")
    plt.legend()
    plt.show()

compare_functions()