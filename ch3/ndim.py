import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

print(A)
print(A.shape)

B = np.array([
    [5, 6],
    [7, 8]
])

print(B)
print(B.shape)

C = np.dot(A, B)
D = A * B

print("np.dot : {0}".format(C))
print("* : {0}".format(D))