# -*- coding: UTF-8 -*-
import pickle
import pandas as pd
import numpy as np
import jieba
'''
这个文件用来构建8182个专利的名称的one-hot矩阵，
用经过jieba分词后的这8182个专利的名称构建的词典来计算，不是单个汉字而是词语
'''
data = pd.read_pickle('data_directory\patent_data_sorted_8182.pkl')
patent_name = [i for i in data['patent_name']]
output = open('data_directory\\patent_name_word_dict_using_jieba_sorted_8182.pkl', 'rb')
dicts = pickle.load(output)
output.close()


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
for i in patent_name:
    seg_list = jieba.cut(i)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    name_vectors.append(trans_article(training_ci))

output = open('data_directory\\patent_name_word_vectors_using_jieba_one_hot_without_padding_sorted_8182.pkl', 'wb')
pickle.dump(name_vectors, output)
output.close()
name_vectors = np.array(name_vectors)
name_vectors_padding = numpy_padding(name_vectors)
print(name_vectors_padding.shape)
output = open('data_directory\\patent_name_word_vectors_using_jieba_one_hot_padding_sorted_8182.pkl', 'wb')
pickle.dump(name_vectors_padding, output)
output.close()
