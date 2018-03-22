import sys, os
sys.path.append(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_batch, t_batch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | {0}, {1}".format(train_acc, test_acc))

# グラフの描画（損失関数の値も表示するよう改善）
x = np.arange(len(train_acc_list))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# １軸（正確性）
ax1.plot(x, train_acc_list, label="train acc")
ax1.plot(x, test_acc_list, label="test acc", linestyle='--')
ax1.set_ylabel('accuracy')
ax1.set_ylim(0, 1.0)

# ２軸（損失関数）
ax2.plot(x, train_loss_list, label="train loss",)
ax2.set_ylabel('loss')
ax2.set_ylim(0)

# x軸
ax1.set_xlabel('epoch')

plt.show()