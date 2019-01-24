# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import math
import chardet
import tensorflow as tf
import numpy as np
'''
通过VGG16网络的方式预测，数据集是8182条的专利名称
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
test_set_x = np.array(test_set_x)
test_set_x = test_set_x.reshape((1182, 40, 128, 1))
# TODO

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


batch_size = 500
total_batch = math.ceil(len(training_set_x) / batch_size)

x = tf.placeholder(tf.float32, [None, 40, 128, 1], name='QueryBatch')
y = tf.placeholder(tf.float32, [None, 1], name='PriceBatch')
x_image = tf.reshape(x, [-1, 40, 128, 1])

W_conv1_1 = weight_variable([3, 3, 1, 64])
b_conv1_1 = bias_variable([64])
h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
W_conv1_2 = weight_variable([3, 3, 64, 64])
b_conv1_2 = bias_variable([64])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)
h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2_1 = weight_variable([3, 3, 64, 128])
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)
W_conv2_2 = weight_variable([3, 3, 128, 128])
b_conv2_2 = bias_variable([128])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)
h_pool2 = max_pool_2x2(h_conv2_2)

W_conv3_1 = weight_variable([3, 3, 128, 256])
b_conv3_1 = bias_variable([256])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)
W_conv3_2 = weight_variable([3, 3, 256, 256])
b_conv3_2 = bias_variable([256])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)
h_pool3 = max_pool_2x2(h_conv3_2)

W_conv4_1 = weight_variable([3, 3, 256, 512])
b_conv4_1 = bias_variable([512])
h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1)
W_conv4_2 = weight_variable([3, 3, 512, 512])
b_conv4_2 = bias_variable([512])
h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)
h_pool4 = max_pool_2x2(h_conv4_2)

W_conv5_1 = weight_variable([3, 3, 512, 512])
b_conv5_1 = bias_variable([512])
h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1)
W_conv5_2 = weight_variable([3, 3, 512, 512])
b_conv5_2 = bias_variable([512])
h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)
h_pool5 = max_pool_2x2(h_conv5_2)

shp = h_pool5.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(h_pool5, [-1, flattened_shape], name="resh1")

fc_6 = 4096
l6_par_range = np.sqrt(6.0 / (flattened_shape + fc_6))
# fc_weight_6 = tf.Variable(tf.random_uniform([flattened_shape, fc_6], - l6_par_range, l6_par_range))
# fc_bias_6 = tf.Variable(tf.random_uniform([fc_6], - l6_par_range, l6_par_range))
fc_weight_6 = weight_variable([flattened_shape, fc_6])
fc_bias_6 = bias_variable([fc_6])
fc_layer_6 = tf.matmul(resh1, fc_weight_6) + fc_bias_6
fc_layer_6_out = tf.nn.relu(fc_layer_6)

fc_7 = 4096
l7_par_range = np.sqrt(6.0 / (fc_6 + fc_7))
# fc_weight_7 = tf.Variable(tf.random_uniform([fc_6, fc_7], - l7_par_range, l7_par_range))
# fc_bias_7 = tf.Variable(tf.random_uniform([fc_7], - l7_par_range, l7_par_range))
fc_weight_7 = weight_variable([fc_6, fc_7])
fc_bias_7 = bias_variable([fc_7])
fc_layer_7 = tf.matmul(fc_layer_6_out, fc_weight_7) + fc_bias_7
fc_layer_7_out = tf.nn.relu(fc_layer_7)

fc_8 = 1
l8_par_range = np.sqrt(6.0 / (fc_7 + fc_8))
# fc_weight_8 = tf.Variable(tf.random_uniform([fc_7, fc_8], - l8_par_range, l8_par_range))
# fc_bias_8 = tf.Variable(tf.random_uniform([fc_8], - l8_par_range, l8_par_range))
fc_weight_8 = weight_variable([fc_7, fc_8])
fc_bias_8 = bias_variable([fc_8])
fc_layer_8 = tf.matmul(fc_layer_7_out, fc_weight_8) + fc_bias_8
fc_layer_8_out = tf.nn.relu(fc_layer_8)

y_conv = tf.reshape(fc_layer_8_out, [-1, 1], name="Prediction")
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
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        if epoch % display_step == 0:
            predict = sess.run(y_conv, feed_dict={x: test_set_x})
            predict = np.array(predict)
            predict = predict.reshape(len(predict))
            test_set_y = np.array(test_set_y)
            test_set_y = test_set_y.reshape(len(test_set_y))
            aape = 0.0
            for i in range(len(predict)):
                aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
            print(aape/len(predict))
        saver.save(sess, savedir + "vgg16_name_8182.cpkt", global_step=epoch)