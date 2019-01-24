# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
import collections
import jieba
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
'''
这个文件用于读取8182个专利的名称，然后使用jieba分词软件进行分词，并且根据分出来的词汇建立一个字典
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


X_train, X_test, y_train, y_test = train_test_split(name_vectors_padding, price_list, test_size=0.1,
                                                    random_state=1)
forest = RandomForestRegressor(n_estimators=5000, criterion='mse',
                               random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print(y_test_pred)
aape = 0.0
for i in range(len(y_test_pred)):
    aape += (abs(y_test_pred[i] - y_test[i][0]))/y_test[i][0]
aape = aape/len(y_test_pred)
print(aape)
# print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
#                                        r2_score(y_test, y_test_pred)))
