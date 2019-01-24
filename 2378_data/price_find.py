# -*- coding: utf-8 -*-
import pickle

output = open('2378_data_directory\word2vec-64dim.pkl', 'rb')
one_hot = pickle.load(output)
output.close()
