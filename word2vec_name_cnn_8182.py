# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import math
import chardet
import tensorflow as tf
import numpy as np
'''
通过cnn+word2vec的方式预测，数据集是8182条的专利名称
'''
pkl_file = open('data_directory\word2vec_name_128dim_8182_2.pkl', 'rb')
word_vectors = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('data_directory\\name_vectors_one_hot_padding_8182.pkl', 'rb')
one_hot_padding = pickle.load(pkl_file)
pkl_file.close()
data = pd.read_pickle('data_directory\patent_data_8182.pkl')
price_list = [[i] for i in data['patent_trade_price']]
empty_vec = [0 for i in range(128)]


def trans_tokens_to_matrix(tokens):
    matrix = list()
    for i in tokens:
        i = int(i)
        if i == 0:
            matrix.append(empty_vec)
        else:
            matrix.append(word_vectors[i])
    return np.array(matrix)


content_matrix = list()
for i in one_hot_padding:
    content_matrix.append(trans_tokens_to_matrix(i))
content_matrix = np.array(content_matrix)
content_matrix = content_matrix.reshape((8182, 40, 128))

# TODO determination of training set is not finished yet
training_set_x = content_matrix[:7000]
training_set_y = price_list[:7000]
test_set_x = content_matrix[7000:]
test_set_y = price_list[7000:]
# TODO

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')


def max_pool_5x4(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 4, 1], strides=[1, 5, 4, 1], padding='SAME')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


batch_size = 500
total_batch = math.ceil(len(training_set_x) / batch_size)

x = tf.placeholder(tf.float32, [None, 40, 128, 1], name='QueryBatch')
y = tf.placeholder(tf.float32, [None, 1], name='PriceBatch')
x_image = tf.reshape(x, [-1, 40, 128, 1])

W_conv1 = weight_variable([4, 4, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1x4(h_conv1)

W_conv2 = weight_variable([4, 4, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_5x4(h_conv2)

W_conv3 = weight_variable([4, 4, 16, 8])
b_conv3 = bias_variable([8])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_4x4(h_conv3)

W_conv4 = weight_variable([4, 4, 8, 1])
b_conv4 = bias_variable([1])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = avg_pool_2x2(h_conv4)


y_conv = tf.reshape(h_pool4, [-1, 1], name="Prediction")
loss = tf.abs(tf.div(tf.subtract(y_conv, y), y))
cost = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
epoch_num = 2000
saver = tf.train.Saver(max_to_keep=1)
savedir = "model\\"
display_step = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):  # 20000
        avg_cost = 0.0
        for k in range(total_batch):
            train_set_x1 = training_set_x[k * batch_size: (k + 1) * batch_size]
            train_set_y1 = training_set_y[k * batch_size: (k + 1) * batch_size]
            train_set_x1 = np.array(train_set_x1)
            train_set_y1 = np.array(train_set_y1)
            train_set_x1 = train_set_x1.reshape((batch_size, 40, 128, 1))
            train_set_y1 = train_set_y1.reshape((batch_size, 1))
            t_, c = sess.run([train_step, cost], feed_dict={x: train_set_x1, y: train_set_y1})
            avg_cost += c / total_batch
            print(epoch, k)

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        saver.save(sess, savedir + "word2vec_cnn_name_8182.cpkt", global_step=epoch)
    test_set_x = np.array(test_set_x)
    test_set_x = test_set_x.reshape((len(test_set_x), 40, 128, 1))
    predict = sess.run(y_conv, feed_dict={x: test_set_x})
    predict = np.array(predict)
    predict = predict.reshape(len(predict))
    test_set_y = np.array(test_set_y)
    test_set_y = test_set_y.reshape(len(test_set_y))
    aape = 0.0
    for i in range(len(predict)):
        aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
    print(aape/len(predict))
