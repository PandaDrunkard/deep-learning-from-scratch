import numpy as np

def OR(x1, x2):
    """
    x1 x2 | y
    ----------
    0  0  | 0
    0  1  | 1
    1  0  | 1
    1  1  | 1
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

m = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

for x in m:
    x1 = x[0]
    x2 = x[1]
    print("{0} OR {1} is {2}".format(x1, x2, OR(x1, x2)))
