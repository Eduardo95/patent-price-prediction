# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import math
import chardet
import tensorflow as tf
import numpy as np
pkl_file = open('2378_data_directory\word2vec-64dim.pkl', 'rb')
word_vectors = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('2378_data_directory\myfile.pkl', 'rb')
dicts = pickle.load(pkl_file)
pkl_file.close()
df_new = pd.read_pickle('2378_data_directory\df.pkl')
reversed_dictionary = dict(zip(dicts.values(), dicts.keys()))
price_list = [[i] for i in df_new['price_list']]
content_list = [i for i in df_new['content_list']]
name_list = [i for i in df_new['name_list']]
empty_vec = [0 for i in range(64)]
empty_fea = [0 for i in range(319)]


def get_token_dict_uni_gram(text):
    token_list = list()
    for i in range(len(text)):
        token_list.append(text[i])
    return token_list


def trans_article(text):
    token_list = get_token_dict_uni_gram(text)
    one_hot = list()
    for token in token_list:
        one_hot.append(dicts[token])
    one_hot.extend(empty_fea)
    return one_hot


def trans_tokens_to_matrix(tokens):
    matrix = list()
    for i in tokens:
        if i == 0:
            matrix.append(empty_vec)
        else:
            matrix.append(word_vectors[i])
    for i in range(328):
        matrix.append(empty_vec)
    return matrix


def numpy_padding(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


content_vectors = list()
for i in range(len(content_list)):
    content_vectors.append(trans_article(content_list[i]))
content_vectors = np.array(content_vectors)
content_vectors = numpy_padding(content_vectors)
print(content_vectors.shape)

content_matrix = list()
for i in range(len(content_vectors)):
    vec = content_vectors[i]
    vec = np.array(vec).reshape(172, 172)
    content_matrix.append(vec)
content_matrix = np.array(content_matrix)
print(content_matrix.shape)

training_set_x = content_matrix[:2000]
training_set_y = price_list[:2000]
test_set_x = content_matrix[2000:]
test_set_y = price_list[2000:]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, w, h):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, w, h, 1], padding='SAME')


def avg_pool(x, w, h):
    return tf.nn.avg_pool(x, ksize=[1, w, h, 1], strides=[1, w, h, 1], padding='SAME')


batch_size = 4
total_batch = math.ceil(len(training_set_x) / batch_size)

x = tf.placeholder(tf.float32, [None, 172, 172, 1], name='QueryBatch')
y = tf.placeholder(tf.float32, [None, 1], name='PriceBatch')
x_image = tf.reshape(x, [-1, 172, 172, 1])

W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 2, 2)

W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2, 2)

shp = h_pool2.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(h_pool2, [-1, flattened_shape], name="resh1")

fc_6 = 2048
fc_weight_6 = weight_variable([flattened_shape, fc_6])
fc_bias_6 = bias_variable([fc_6])
fc_layer_6 = tf.matmul(resh1, fc_weight_6) + fc_bias_6
fc_layer_6_out = tf.nn.relu(fc_layer_6)

fc_7 = 1
fc_weight_7 = weight_variable([fc_6, fc_7])
fc_bias_7 = bias_variable([fc_7])
fc_layer_7 = tf.matmul(fc_layer_6_out, fc_weight_7) + fc_bias_7
fc_layer_7_out = tf.nn.relu(fc_layer_7)

y_conv = tf.reshape(fc_layer_7_out, [-1, 1], name="Prediction")
loss = tf.abs(tf.div(tf.subtract(y_conv, y), y))
cost = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
epoch_num = 2000
saver = tf.train.Saver(max_to_keep=1)
savedir = "log2\\"
display_step = 200
list_x = list()
list_y = list()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):  # 20000
        avg_cost = 0.0
        for k in range(total_batch):
            train_set_x1 = training_set_x[k * batch_size: (k + 1) * batch_size]
            train_set_y1 = training_set_y[k * batch_size: (k + 1) * batch_size]
            train_set_x1 = np.array(train_set_x1)
            train_set_y1 = np.array(train_set_y1)
            train_set_x1 = train_set_x1.reshape((batch_size, 172, 172, 1))
            train_set_y1 = train_set_y1.reshape((batch_size, 1))
            t_, c = sess.run([train_step, cost], feed_dict={x: train_set_x1, y: train_set_y1})
            avg_cost += c / total_batch
            print(epoch, k)

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            list_x.append(epoch + 1)
            list_y.append(avg_cost)
        saver.save(sess, savedir + "lenet.cpkt", global_step=epoch)

    predict = sess.run(y_conv, feed_dict={x: test_set_x})
    print(predict)
    print(test_set_y)