# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import math
import chardet
from collections import Counter
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
pkl_file = open('2378_data_directory\word2vec-64dim.pkl', 'rb')
word_vectors = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('2378_data_directory\myfile.pkl', 'rb')
dicts = pickle.load(pkl_file)
pkl_file.close()
df_new = pd.read_pickle('2378_data_directory\df.pkl')
# print(df_new)
df_new = df_new.sort_values("price_list", ascending=False)
# print(df_new1)
price_list = [[i] for i in df_new['price_list']]
content_list = [i for i in df_new['content_list']]
name_list = [i for i in df_new['name_list']]


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
    return one_hot


def numpy_padding(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def trans_tokens_to_matrix(tokens):
    matrix = list()
    for i in tokens:
        i = int(i)
        if i == 0:
            matrix.append([0 for i in range(64)])
        else:
            matrix.append(word_vectors[i])
    return np.array(matrix)


content_vectors = list()
for i in range(len(content_list)):
    content_vectors.append(trans_article(content_list[i]))
content_vectors = np.array(content_vectors)
print(content_vectors.shape)
# content_vectors = numpy_padding(content_vectors)
# print(content_vectors.shape)

content_matrix = list()
for i in content_vectors:
    content_matrix.append(trans_tokens_to_matrix(i))
content_matrix = np.array(content_matrix)

avg_word_vec = list()
for i in content_matrix:
    if len(i) == 0:
        b = [0 for k in range(64)]
    else:
        b = i[0]
        for j in range(len(i)-1):
            b += i[j+1]
        b /= len(i)
    avg_word_vec.append(b)
avg_word_vec = np.array(avg_word_vec)
dim = 64
X_train_vec, X_test_vec, y_train_price, y_test_price = train_test_split(avg_word_vec, price_list, test_size=0.1, random_state=1)
# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
y_train_class = list()
for i in y_train_price:
    if i[0] > 1:
        y_train_class.append(2)
    elif 1 >= i[0] >= 0.5:
        y_train_class.append(1)
    else:
        y_train_class.append(0)
forest = RandomForestClassifier(n_estimators=5000, random_state=1, n_jobs=-1, max_depth=100)
forest.fit(X_train_vec, y_train_class)
y_test_class = list()
for i in y_test_price:
    if i[0] > 1:
        y_test_class.append(2)
    elif 1 >= i[0] >= 0.5:
        y_test_class.append(1)
    else:
        y_test_class.append(0)
print(len(X_train_vec), len(y_train_price), len(y_train_class))
'''
使用分类器分类结果
'''
y_test_class_pred = forest.predict(X_test_vec)
not_equ = 0
for i in range(len(y_test_class)):
    if y_test_class[i] != y_test_class_pred[i]:
        not_equ += 1
print(len(X_test_vec), len(y_test_price), len(y_test_class), len(y_test_class_pred))
'''
网络结构
'''
print("----------train begin------------")
query_batch = tf.placeholder(tf.float32, shape=[None, dim], name='QueryBatch')
price_batch = tf.placeholder(tf.float32, shape=[None, 1], name='PriceBatch')


L1_N = 300
l1_par_range = np.sqrt(6.0 / (dim + L1_N))
weight1 = tf.Variable(tf.random_uniform([dim, L1_N], - l1_par_range, l1_par_range))
bias1 = tf.Variable(tf.random_uniform([L1_N], - l1_par_range, l1_par_range))
query_l1 = tf.matmul(query_batch, weight1) + bias1
query_l1_out = tf.nn.relu(query_l1)
query_l1_out = tf.nn.dropout(query_l1_out, keep_prob=0.6)

L2_N = 300
l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], - l2_par_range, l2_par_range))
bias2 = tf.Variable(tf.random_uniform([L2_N], - l2_par_range, l2_par_range))
query_l2 = tf.matmul(query_l1_out, weight2) + bias2
query_l2_out = tf.nn.relu(query_l2)
query_l2_out = tf.nn.dropout(query_l2_out, keep_prob=0.6)

L3_N = 128
l3_par_range = np.sqrt(6.0 / (L2_N + L3_N))
weight3 = tf.Variable(tf.random_uniform([L2_N, L3_N], - l3_par_range, l3_par_range))
bias3 = tf.Variable(tf.random_uniform([L3_N], - l3_par_range, l3_par_range))
query_l3 = tf.matmul(query_l2_out, weight3) + bias3
query_l3_out = tf.nn.relu(query_l3)
query_l3_out = tf.nn.dropout(query_l3_out, keep_prob=0.6)

L4_N = 1
l4_par_range = np.sqrt(6.0 / (L3_N + L4_N))
weight4 = tf.Variable(tf.random_uniform([L3_N, L4_N], - l4_par_range, l4_par_range))
bias4 = tf.Variable(tf.random_uniform([L4_N], - l4_par_range, l4_par_range))
query_l4 = tf.add(tf.matmul(query_l3_out, weight4), bias4, name="Prediction")
query_l4_out = tf.nn.dropout(query_l4, keep_prob=0.6)
loss = tf.abs(tf.div(tf.subtract(query_l4, price_batch), price_batch))
cost = tf.reduce_mean(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
training_epochs = 5000
display_step = 200
saver = tf.train.Saver()
savedir = "model/"

'''对不同类别做回归'''
price_0_x_train = list()
price_0_y_train = list()
price_1_x_train = list()
price_1_y_train = list()
price_2_x_train = list()
price_2_y_train = list()
for i in range(len(X_train_vec)):
    if y_train_class[i] == 0:
        price_0_x_train.append(X_train_vec[i])
        price_0_y_train.append(y_train_price[i])
    elif y_train_class[i] == 1:
        price_1_x_train.append(X_train_vec[i])
        price_1_y_train.append(y_train_price[i])
    else:
        price_2_x_train.append(X_train_vec[i])
        price_2_y_train.append(y_train_price[i])

pred_class_0_x_test = list()
pred_class_0_y_test = list()
pred_class_1_x_test = list()
pred_class_1_y_test = list()
pred_class_2_x_test = list()
pred_class_2_y_test = list()

for i in range(len(X_test_vec)):
    if y_test_class_pred[i] == 0:
        pred_class_0_x_test.append(X_test_vec[i])
        pred_class_0_y_test.append(y_test_price[i])
    elif y_test_class_pred[i] == 1:
        pred_class_1_x_test.append(X_test_vec[i])
        pred_class_1_y_test.append(y_test_price[i])
    else:
        pred_class_2_x_test.append(X_test_vec[i])
        pred_class_2_y_test.append(y_test_price[i])
aape_total = 0.0
test_total = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([train_step, cost], feed_dict={query_batch: price_0_x_train, price_batch: price_0_y_train})
        avg_cost = c / 1
        predict = sess.run(query_l4, feed_dict={query_batch: pred_class_0_x_test})
        predict = np.array(predict)
        predict = predict.reshape(len(predict))
        predict = (predict)
        test_set_y = np.array(pred_class_0_y_test)
        test_set_y = test_set_y.reshape(len(test_set_y))
        aape = 0.0
        for i in range(len(predict)):
            aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
        aape_total += aape
        test_total += len(predict)
        print("aape:", aape / len(predict))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([train_step, cost], feed_dict={query_batch: price_1_x_train, price_batch: price_1_y_train})
        avg_cost = c / 1
        predict = sess.run(query_l4, feed_dict={query_batch: pred_class_1_x_test})
        predict = np.array(predict)
        predict = predict.reshape(len(predict))
        predict = (predict)
        test_set_y = np.array(pred_class_1_y_test)
        test_set_y = test_set_y.reshape(len(test_set_y))
        aape = 0.0
        for i in range(len(predict)):
            aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
        aape_total += aape
        test_total += len(predict)
        print("aape:", aape / len(predict))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([train_step, cost], feed_dict={query_batch: price_2_x_train, price_batch: price_2_y_train})
        avg_cost = c / 1
        predict = sess.run(query_l4, feed_dict={query_batch: pred_class_2_x_test})
        predict = np.array(predict)
        predict = predict.reshape(len(predict))
        predict = (predict)
        test_set_y = np.array(pred_class_2_y_test)
        test_set_y = test_set_y.reshape(len(test_set_y))
        aape = 0.0
        for i in range(len(predict)):
            aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
        aape_total += aape
        test_total += len(predict)
        print("aape:", aape / len(predict))

print("total_aape", aape_total / test_total)