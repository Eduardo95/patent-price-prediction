# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
import collections
import jieba
'''
这个文件用于读取2185个专利的具体内容，然后使用jieba分词软件进行分词，并且根据分出来的词汇建立一个字典
'''
data = pd.read_pickle('2185_data_directory\patent_data_2185.pkl')
content_list = [i for i in data['content_list']]
tokens = list()
for i in content_list:
    seg_list = jieba.cut(i)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    # 以空格将字符串分开
    tokens.extend(training_ci)


def build_dataset(token_list, num_tokens):
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
output = open('2185_data_directory\\content_dict_jieba_2185.pkl', 'wb')
pickle.dump(dicts, output)
output.close()
