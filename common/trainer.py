import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.optimizers import *

class Trainer(object):
    """
    NNの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.optimizer = self.__get_optimizer(optimizer, optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        
        # runtime variables
        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def __get_optimizer(self, optimizer, param):
        optimizer_class_dict = {
            'sgd': SGD,
            'momentum': Momentum,
            'addgrad': AddGrad,
            'rmsprop': RMSprop,
            'adam': Adam
        }
        return optimizer_class_dict[optimizer.lower()](**param)

    def __print(self, value):
        if self.verbose: print(value)

    def train(self):
        # train network
        for i in range(self.max_iter):
            self.__train_step()
        
        # output final accuracy
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        self.__print("=============== Final Test Accuracy ===============")
        self.__print("test acc:" + str(test_acc))

    
    def __train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        self.__print("train loss:" + str(loss))

        self.__output_status_per_epoch()

        
        self.current_iter += 1

    def __output_status_per_epoch(self):
        if self.__new_epoch_starts():
            self.current_epoch += 1

            x_train_sample, t_train_sample, x_test_sample, t_test_sample = \
                self.__get_evaluate_samples()
            
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            self.__print('=== epoch:{0}, train acc:{1}, test acc:{2} ==='.format(
                self.current_epoch, train_acc, test_acc
            ))

    def __new_epoch_starts(self):
        return self.current_iter % self.iter_per_epoch == 0
        
    def __get_evaluate_samples(self):
        if self.evaluate_sample_num_per_epoch is None:
            return self.x_train, self.t_train, self.x_test, self.t_test
        else:
            t = self.evaluate_sample_num_per_epoch
            return self.x_train[:t], self.t_train[:t], self.x_test[:t], self.t_test[:t]
        