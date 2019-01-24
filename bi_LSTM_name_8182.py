# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
'''
通过Bi-LSTM的方式预测，数据集是8182条的专利名称
'''
pkl_file = open('data_directory\\patent_name_word_vectors_using_jieba_one_hot_padding_8182.pkl', 'rb')
one_hot_padding = pickle.load(pkl_file)
pkl_file.close()
data = pd.read_pickle('data_directory\patent_data_8182.pkl')
price_list = [[i] for i in data['patent_trade_price']]
print(one_hot_padding.shape)
dim = 18


training_set_x1 = one_hot_padding[:7000]
training_set_y1 = price_list[:7000]
test_set_x = one_hot_padding[7000:]
test_set_y = price_list[7000:]

# 参数设置
learning_rate = 0.001
training_epochs = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 1  # MNIST data 输入 (img shape: 28*28)
n_steps = 18  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 1  # MNIST 列别 (0-9 ，一共10类)

tf.reset_default_graph()

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

x1 = tf.unstack(x, n_steps, 1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# 反向cell
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
print(len(outputs), outputs[0].shape, outputs[1].shape)
outputs = tf.concat(outputs, 2)
outputs = tf.transpose(outputs, [1, 0, 2])

pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

loss = tf.abs(tf.div(tf.subtract(pred, y), y))
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# 启动session
training_set_x = np.reshape(training_set_x1, (len(training_set_x1), n_steps, n_input))
training_set_y = np.reshape(training_set_y1, (len(training_set_y1), n_classes))
test_set_x = np.reshape(test_set_x, (len(test_set_x), n_steps, n_input))
test_set_y = np.reshape(test_set_y, (len(test_set_y), n_classes))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: training_set_x, y: training_set_y})
        avg_cost = c / 1
        predict = sess.run(pred, feed_dict={x: test_set_x})
        predict = np.array(predict)
        predict = predict.reshape(len(predict))
        test_set_y = np.array(test_set_y)
        test_set_y = test_set_y.reshape(len(test_set_y))
        aape = 0.0
        for i in range(len(predict)):
            aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
        print("aape:", aape / len(predict))
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
