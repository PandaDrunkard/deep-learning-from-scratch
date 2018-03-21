import sys, os
sys.path.append(os.pardir)
from common.layer import *
import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
print(x[x <= 0])
