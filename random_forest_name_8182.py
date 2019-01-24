# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
'''
通过随机森林的方式预测，数据集是8182条的专利名称
'''
pkl_file = open('data_directory\\patent_name_word_vectors_using_jieba_one_hot_padding_8182.pkl', 'rb')
one_hot_padding = pickle.load(pkl_file)
pkl_file.close()
data = pd.read_pickle('data_directory\patent_data_8182.pkl')
price_list = [[i] for i in data['patent_trade_price']]
print(one_hot_padding.shape)
dim = 18
price_list = np.array(price_list)
print(price_list.shape)

X_train, X_test, y_train, y_test = train_test_split(one_hot_padding, price_list, test_size=0.1,
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
