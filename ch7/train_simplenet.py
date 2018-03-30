import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from common.networks import SimpleCNN
from common.trainer import Trainer
import matplotlib.pyplot as plt

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# trainer = Trainer(network, x_train, t_train, x_test, t_test)

# trainer.train()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleCNN()

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.01},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(20)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()