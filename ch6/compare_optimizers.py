import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.networks import TwoLayerNet
from common.optimizers import SGD, Momentum
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class TestEntity():
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer
        self.test_acc_list = []
        self.train_acc_list = []
        self.train_loss_list = []


(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

optimizers = {}
optimizers['SGD'] = TestEntity(TwoLayerNet(input_size=784, hidden_size=50, output_size=10), SGD())
optimizers['Momentum'] = TestEntity(TwoLayerNet(input_size=784, hidden_size=50, output_size=10), Momentum())

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(train_size / batch_size, 1)

epochs = 0

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for opt_name in optimizers.keys():
        target = optimizers[opt_name]
        network = target.network
        optimizer = target.optimizer

        grads = network.gradient(x_batch, t_batch)
        params = network.params

        optimizer.update(params, grads)

        if i % iter_per_epoch == 0:
            train_loss = network.loss(x_batch, t_batch)
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            target.train_loss_list.append(train_loss)
            target.train_acc_list.append(train_acc)
            target.test_acc_list.append(test_acc)
            print("[{2}] train acc, test acc, train loss | {0}, {1}".format(train_acc, test_acc, opt_name))
            epochs += 1

epochs /= len(optimizers.keys())

# グラフの描画（損失関数の値も表示するよう改善）
x = np.arange(int(epochs))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
plots = []
labels = []

# １軸（正確性）
for opt_name in optimizers.keys():
    ax1.plot(x, optimizers[opt_name].test_acc_list, label=opt_name)
ax1.set_ylabel('accuracy')
ax1.set_ylim(0, 1.0)
ax1.legend(loc='center right')

# ２軸（損失関数）
for opt_name in optimizers.keys():
    p = ax2.plot(x, optimizers[opt_name].train_loss_list, linestyle='dashed')
    plots.append(p)
    labels.append(opt_name)
ax2.set_ylabel('loss')
ax2.set_ylim(0)

# x軸
ax1.set_xlabel('epoch')

plt.show()