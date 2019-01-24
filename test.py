# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
import collections
import jieba
import numpy as np
import tensorflow as tf
'''
这个文件用于读取8182个专利的名称，然后使用jieba分词软件进行分词，
然后去掉一些出现频率比较高的词，例如“的”等，并且根据分出来的词
汇建立一个字典，最后对价格非常低的和价格非常高的样本进行重采样，
然后使用gru进行分类。
Fuck it，没什么用。
'''
data = pd.read_pickle('data_directory\patent_data_sorted_8182.pkl')
patent_name = [i for i in data['patent_name']]
price_list = [[i] for i in data['patent_trade_price']]
dim = 17
tokens = list()
names_by_token = list()
for i in patent_name:
    seg_list = jieba.cut(i)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    names_by_token.append(training_ci)
    tokens.extend(training_ci)


def build_dataset(token_list, num_tokens):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(token_list).most_common(num_tokens - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in token_list:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


training_label, count, dicts, words = build_dataset(tokens, len(tokens))
names_by_token_drop_frequent_noice = list()
for i in names_by_token:
    for j in range(13):
        if words[j] in i:
            i.remove(words[j])
    for j in range(13):
        if words[j] in i:
            i.remove(words[j])
    if '、' in i:
        i.remove('、')
    if '式' in i:
        i.remove('式')
    names_by_token_drop_frequent_noice.append(i)
print(names_by_token_drop_frequent_noice)
# output = open('data_directory\\patent_name_word_dict_using_jieba_sorted_8182.pkl', 'wb')
# pickle.dump(dicts, output)
# output.close()


def trans_article(token_list):
    one_hot = list()
    for token in token_list:
        one_hot.append(dicts[token])
    return one_hot


def numpy_padding(data):
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


name_vectors = list()
for i in names_by_token_drop_frequent_noice:
    name_vectors.append(trans_article(i))
print(name_vectors)
name_vectors = np.array(name_vectors)
name_vectors_padding = numpy_padding(name_vectors)
print(name_vectors_padding.shape)


price_low = name_vectors_padding[:586]
price_high = name_vectors_padding[7946:]
price_middie = name_vectors_padding[586:7946]

price_low_y = price_list[:586]
price_high_y = price_list[7946:]
price_middle_y = price_list[586:7946]
data_set_x = list()
data_set_y = list()
for i in range(10):
    data_set_x.extend(price_low)
    data_set_y.extend(price_low_y)
data_set_x.extend(price_middie)
data_set_y.extend(price_middle_y)
for i in range(10):
    data_set_x.extend(price_high)
    data_set_y.extend(price_high_y)

data_set_x = np.array(data_set_x)
data_set_y = np.array(data_set_y)
print(data_set_x.shape)
print(data_set_y.shape)
training_set_x1 = data_set_x[:15000]
training_set_y1 = data_set_y[:15000]
test_set_x = data_set_x[15000:]
test_set_y = data_set_y[15000:]
'''
Below is a basic gru-cell RNN model
'''
n_input = 1  #
n_steps = 18  # 一个文本是一串序列
n_hidden = 128  # hidden layer num of features
n_classes = 1  # 最后的预测值
tf.reset_default_graph()

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
x1 = tf.unstack(x, n_steps, 1)
gru = tf.contrib.rnn.GRUCell(n_hidden)
outputs = tf.contrib.rnn.static_rnn(gru, x1, dtype=tf.float32)

pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

learning_rate = 0.0001
training_epochs = 10000
# Define loss and optimizer
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