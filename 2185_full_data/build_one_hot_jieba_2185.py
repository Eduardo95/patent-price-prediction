# -*- coding: UTF-8 -*-
import pickle
import pandas as pd
import numpy as np
import jieba
'''
这个文件用来构建2185个专利的内容的one-hot矩阵，
用经过jieba分词后的这2185个专利的内容构建的词典来计算，不是单个汉字而是词语
'''
data = pd.read_pickle('2185_data_directory\patent_data_2185.pkl')
content_list = [i for i in data['content_list']]
output = open('2185_data_directory\content_dict_jieba_2185.pkl', 'rb')
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


content_one_hot = list()
for i in content_list:
    seg_list = jieba.cut(i)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    content_one_hot.append(trans_article(training_ci))

output = open('2185_data_directory\content_one_hot_jieba_2185.pkl', 'wb')
pickle.dump(content_one_hot, output)
output.close()
content_one_hot = np.array(content_one_hot)
content_one_hot_padding = numpy_padding(content_one_hot)
print(content_one_hot_padding.shape)
output = open('2185_data_directory\content_one_hot_jieba_padding_2185.pkl', 'wb')
pickle.dump(content_one_hot_padding, output)
output.close()
