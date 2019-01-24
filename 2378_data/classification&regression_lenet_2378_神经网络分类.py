# -*- coding: utf-8 -*-
import pickle
import pandas as pd
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
df_new = df_new.sort_values("price_list", ascending=False)
price_list = [[i] for i in df_new['price_list']]
content_list = [i for i in df_new['content_list']]
name_list = [i for i in df_new['name_list']]
empty_vec = [0 for i in range(64)]
empty_fea_train = [0 for i in range(319)]
empty_fea_test = [0 for i in range(14076)]


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
            matrix.append([0 for k in range(64)])
        else:
            matrix.append(word_vectors[i])
    return np.array(matrix)


content_vectors = list()
for i in range(len(content_list)):
    content_vectors.append(trans_article(content_list[i]))
content_vectors = np.array(content_vectors)

X_train_content_vec, X_test_content_vec, y_train_price, y_test_price = train_test_split(content_vectors, price_list,
                                                                                        test_size=0.1, random_state=1)

'''
获取训练集和测试集的172*172文本矩阵，用于LeNet-5 CNN
'''
X_train_content_matrix = list()
X_train_content_vec_padding = numpy_padding(X_train_content_vec)
for i in X_train_content_vec_padding:
    X_train_content_matrix.append(np.append(i, empty_fea_train)[:10000].reshape(100, 100))
X_train_content_matrix = np.array(X_train_content_matrix)
print(X_train_content_matrix.shape)
X_test_content_matrix = list()
X_test_content_vec_padding = numpy_padding(X_test_content_vec)
for i in X_test_content_vec_padding:
    X_test_content_matrix.append(np.append(i, empty_fea_test)[:10000].reshape(100, 100))
X_test_content_matrix = np.array(X_test_content_matrix)
print(X_test_content_matrix.shape)

'''
获取训练集和测试集的平均词向量，用于分类
'''
X_train_word2vec = list()
for i in X_train_content_vec:
    X_train_word2vec.append(trans_tokens_to_matrix(i))
X_train_word2vec = np.array(X_train_word2vec)
X_train_avg_word_vec = list()
for i in X_train_word2vec:
    if len(i) == 0:
        b = [0 for k in range(64)]
    else:
        b = i[0]
        for j in range(len(i)-1):
            b += i[j+1]
        b /= len(i)
    X_train_avg_word_vec.append(b)
X_train_avg_word_vec = np.array(X_train_avg_word_vec)

X_test_word2vec = list()
for i in X_test_content_vec:
    X_test_word2vec.append(trans_tokens_to_matrix(i))
X_test_word2vec = np.array(X_test_word2vec)
X_test_avg_word_vec = list()
for i in X_test_word2vec:
    if len(i) == 0:
        b = [0 for k in range(64)]
    else:
        b = i[0]
        for j in range(len(i)-1):
            b += i[j+1]
        b /= len(i)
    X_test_avg_word_vec.append(b)
X_test_avg_word_vec = np.array(X_test_avg_word_vec)
print(X_train_avg_word_vec.shape, X_test_avg_word_vec.shape)
dim = 64

# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
'''以0、1、2方式标注的标签'''
y_train_class = list()
for i in y_train_price:
    if i[0] > 1:
        y_train_class.append(2)
    elif 1 >= i[0] >= 0.5:
        y_train_class.append(1)
    else:
        y_train_class.append(0)
y_test_class = list()
for i in y_test_price:
    if i[0] > 1:
        y_test_class.append(2)
    elif 1 >= i[0] >= 0.5:
        y_test_class.append(1)
    else:
        y_test_class.append(0)

'''以[1, 0, 0]、[0, 1, 0]、[0, 0, 1]方式标注的标签'''
y_train_class_nn = list()
for i in y_train_class:
    if i == 2:
        y_train_class_nn.append([0, 0, 1])
    elif i == 1:
        y_train_class_nn.append([0, 1, 0])
    else:
        y_train_class_nn.append([1, 0, 0])
y_test_class_nn = list()
for i in y_test_class:
    if i == 2:
        y_test_class_nn.append([0, 0, 1])
    elif i == 1:
        y_test_class_nn.append([0, 1, 0])
    else:
        y_test_class_nn.append([1, 0, 0])

'''
使用分类器分类结果
'''
forest = RandomForestClassifier(n_estimators=5000, random_state=1, n_jobs=-1, max_depth=100)
forest.fit(X_train_avg_word_vec, y_train_class)
y_test_class_pred = forest.predict(X_test_avg_word_vec)

'''神经网络分类器'''
x_classification = tf.placeholder(tf.float32, [None, 64], name='class_x_batch')
y_classification = tf.placeholder(tf.float32, [None, 3], name='class_y_batch')

classification_weight1 = tf.Variable(tf.truncated_normal([64, 256], stddev=0.1))
classification_bias1 = tf.Variable(tf.truncated_normal([256], stddev=0.1))
classification_l1 = tf.matmul(x_classification, classification_weight1) + classification_bias1
classification_l1_out = tf.nn.relu(classification_l1)
classification_l1_out = tf.nn.dropout(classification_l1_out, keep_prob=0.6)

classification_weight2 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.1))
classification_bias2 = tf.Variable(tf.truncated_normal([256], stddev=0.1))
classification_l2 = tf.matmul(classification_l1_out, classification_weight2) + classification_bias2
classification_l2_out = tf.nn.relu(classification_l2)
classification_l2_out = tf.nn.dropout(classification_l2_out, keep_prob=0.6)

classification_weight3 = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1))
classification_bias3 = tf.Variable(tf.truncated_normal([3], stddev=0.1))
classification_lout = tf.matmul(classification_l2_out, classification_weight3) + classification_bias3
classification_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classification_lout,
                                                                             labels=y_classification))
classification_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(classification_cost)
training_epochs = 2000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([classification_optimizer, classification_cost],
                        feed_dict={x_classification: X_train_avg_word_vec, y_classification: y_train_class_nn})
        avg_cost = c / 1
        print(epoch)
    predict_nn = sess.run(classification_lout, feed_dict={x_classification: X_test_avg_word_vec})
    predict = np.array(predict_nn)
    print(predict.shape)
    print(predict)

'''
回归网络结构
'''


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


x = tf.placeholder(tf.float32, [None, 100, 100, 1], name='QueryBatch')
y = tf.placeholder(tf.float32, [None, 1], name='PriceBatch')
x_image = tf.reshape(x, [-1, 100, 100, 1])

W_conv1 = weight_variable([2, 2, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='Conv1')
h_pool1 = avg_pool(h_conv1, 5, 5)

W_conv2 = weight_variable([2, 2, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='Conv2')
h_pool2 = avg_pool(h_conv2, 5, 5)

shp = h_pool2.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(h_pool2, [-1, flattened_shape], name="resh1")

fc_6 = 128
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

'''对不同类别做回归'''
price_0_x_train = list()
price_0_y_train = list()
price_1_x_train = list()
price_1_y_train = list()
price_2_x_train = list()
price_2_y_train = list()
for i in range(len(X_train_content_matrix)):
    if y_train_class[i] == 0:
        price_0_x_train.append(X_train_content_matrix[i])
        price_0_y_train.append(y_train_price[i])
    elif y_train_class[i] == 1:
        price_1_x_train.append(X_train_content_matrix[i])
        price_1_y_train.append(y_train_price[i])
    else:
        price_2_x_train.append(X_train_content_matrix[i])
        price_2_y_train.append(y_train_price[i])
price_0_x_train = np.array(price_0_x_train).reshape(len(price_0_x_train), 100, 100, 1)
price_1_x_train = np.array(price_1_x_train).reshape(len(price_1_x_train), 100, 100, 1)
price_2_x_train = np.array(price_2_x_train).reshape(len(price_2_x_train), 100, 100, 1)
price_0_y_train = np.array(price_0_y_train).reshape(len(price_0_y_train), 1)
price_1_y_train = np.array(price_1_y_train).reshape(len(price_1_y_train), 1)
price_2_y_train = np.array(price_2_y_train).reshape(len(price_2_y_train), 1)
pred_class_0_x_test = list()
pred_class_0_y_test = list()
pred_class_1_x_test = list()
pred_class_1_y_test = list()
pred_class_2_x_test = list()
pred_class_2_y_test = list()
for i in range(len(X_test_content_matrix)):
    if y_test_class_pred[i] == 0:
        pred_class_0_x_test.append(X_test_content_matrix[i])
        pred_class_0_y_test.append(y_test_price[i])
    elif y_test_class_pred[i] == 1:
        pred_class_1_x_test.append(X_test_content_matrix[i])
        pred_class_1_y_test.append(y_test_price[i])
    else:
        pred_class_2_x_test.append(X_test_content_matrix[i])
        pred_class_2_y_test.append(y_test_price[i])
pred_class_0_x_test = np.array(pred_class_0_x_test).reshape(len(pred_class_0_x_test), 100, 100, 1)
pred_class_1_x_test = np.array(pred_class_1_x_test).reshape(len(pred_class_1_x_test), 100, 100, 1)
pred_class_2_x_test = np.array(pred_class_2_x_test).reshape(len(pred_class_2_x_test), 100, 100, 1)
pred_class_0_y_test = np.array(pred_class_0_y_test).reshape(len(pred_class_0_y_test), 1)
pred_class_1_y_test = np.array(pred_class_1_y_test).reshape(len(pred_class_1_y_test), 1)
pred_class_2_y_test = np.array(pred_class_2_y_test).reshape(len(pred_class_2_y_test), 1)
aape_total = 0.0
test_total = 0
training_epochs = 2000

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(training_epochs):
#         _, c = sess.run([train_step, cost], feed_dict={x: price_0_x_train, y: price_0_y_train})
#         avg_cost = c / 1
#         print(epoch)
#     predict = sess.run(y_conv, feed_dict={x: pred_class_0_x_test})
#     predict = np.array(predict)
#     predict = predict.reshape(len(predict))
#     predict = (predict)
#     test_set_y = np.array(pred_class_0_y_test)
#     test_set_y = test_set_y.reshape(len(test_set_y))
#     aape = 0.0
#     for i in range(len(predict)):
#         aape += abs((predict[i] - test_set_y[i]) / test_set_y[i])
#     aape_total += aape
#     test_total += len(predict)
#     print("aape:", aape / len(predict))
batch_size = 700
total_batch = 3
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for k in range(total_batch - 1):
            train_set_x1 = price_1_x_train[k * batch_size: (k + 1) * batch_size]
            train_set_y1 = price_1_y_train[k * batch_size: (k + 1) * batch_size]
            train_set_x1 = np.array(train_set_x1)
            train_set_y1 = np.array(train_set_y1)
            train_set_x1 = train_set_x1.reshape((batch_size, 100, 100, 1))
            train_set_y1 = train_set_y1.reshape((batch_size, 1))
            _, c = sess.run([train_step, cost], feed_dict={x: train_set_x1, y: train_set_y1})
            avg_cost = c / 1
            print(epoch)
        train_set_x1 = price_1_x_train[2 * batch_size:]
        train_set_y1 = price_1_y_train[2 * batch_size:]
        train_set_x1 = np.array(train_set_x1)
        train_set_y1 = np.array(train_set_y1)
        train_set_x1 = train_set_x1.reshape((len(train_set_x1), 100, 100, 1))
        train_set_y1 = train_set_y1.reshape((len(train_set_y1), 1))
        _, c = sess.run([train_step, cost], feed_dict={x: train_set_x1, y: train_set_y1})
        avg_cost = c / 1
        print(epoch)
    predict = sess.run(y_conv, feed_dict={x: pred_class_1_x_test})
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
        _, c = sess.run([train_step, cost], feed_dict={x: price_2_x_train, y: price_2_y_train})
        avg_cost = c / 1
        print(epoch)
    predict = sess.run(y_conv, feed_dict={x: pred_class_2_x_test})
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
