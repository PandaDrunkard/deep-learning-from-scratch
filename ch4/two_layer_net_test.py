import sys, os
sys.path.append(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

# x = np.random.randn(100,784)
# t = np.argmax(net.predict(x), axis=1)

# print(t)

print('loading mnist data...')

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print('finished')

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print_per = iters_num / 10

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    if i % print_per == 0:
        print_msg = lambda msg: print(msg)
    else:
        print_msg = lambda msg: None

    print_msg('processing {0}/{1}'.format(i, iters_num))

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print_msg('batch determined')

    # very very slow...
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    print_msg('gradient calculated')

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    print_msg('parameters updated')

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print_msg('loss function valuated')

plt.plot(range(iters_num), train_loss_list)
plt.show()