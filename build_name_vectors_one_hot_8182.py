# -*- coding: UTF-8 -*-
import pickle
import pandas as pd
import numpy as np
'''
这个文件用来构建8182个专利的名称的one-hot矩阵，用这8182个专利的名称构建的词典来计算
'''
data = pd.read_pickle('data_directory\patent_data_sorted_8182.pkl')
patent_name = [i for i in data['patent_name']]
output = open('data_directory\\name_dict_sorted_8182.pkl', 'rb')
dicts = pickle.load(output)
output.close()


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
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


name_vectors = list()
for i in range(len(patent_name)):
    name_vectors.append(trans_article(patent_name[i]))
print(name_vectors)
output = open('data_directory\\name_vectors_one_hot_without_padding_sorted_8182.pkl', 'wb')
pickle.dump(name_vectors, output)
output.close()
name_vectors = np.array(name_vectors)
name_vectors_padding = numpy_padding(name_vectors)
print(name_vectors_padding.shape)
output = open('data_directory\\name_vectors_one_hot_padding_sorted_8182.pkl', 'wb')
pickle.dump(name_vectors_padding, output)
output.close()
