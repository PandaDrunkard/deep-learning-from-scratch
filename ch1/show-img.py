import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./sample/dataset/lena.png')
plt.imshow(img)

plt.show()