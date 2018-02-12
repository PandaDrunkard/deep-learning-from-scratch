import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

def show_img(img):
    pass