import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from common.networks import TwoLayerNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

trainer = Trainer(network, x_train, t_train, x_test, t_test)

trainer.train()