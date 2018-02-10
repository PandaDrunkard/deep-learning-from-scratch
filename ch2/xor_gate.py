from and_gate import AND
from or_gate import OR
from nand_gate import NAND

def XOR(x1, x2):
    """
    x1 x2 | y
    ----------
    0  0  | 0
    0  1  | 1
    1  0  | 1
    1  1  | 0
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

m = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

for x in m:
    x1 = x[0]
    x2 = x[1]
    print("{0} XOR {1} is {2}".format(x1, x2, XOR(x1, x2)))
