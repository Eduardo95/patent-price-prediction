# -*- coding: utf-8 -*-
import pandas as pd
import os
import chardet
import xlrd
'''
这个文件用于从2185个txt文件中读取原始内容，并和文件名称一个组成一个dataframe
'''


def read_each_file(file_path):
    file_path_list = list()
    file_name_list = list()
    path_dir = os.listdir(file_path)
    for allDir in path_dir:
        child = os.path.join('%s\%s' % (file_path, allDir))
        file_path_list.append(child)
        file_name_list.append(allDir.strip(".txt"))
        # print(child)
    return file_path_list, file_name_list


# 读取文件内容并打印
def read_file(filename):
    fopen = open(filename, 'rb')  # r 代表read
    result = ""
    for eachLine in fopen:
        fencoding = chardet.detect(eachLine)
        newline = eachLine.decode(fencoding['encoding'])
        result += newline.strip("\r\n")
        # print(newline)
        # print("读取到得内容如下：", eachLine)
    fopen.close()
    return result


filePathC = "C:\\Users\skydata\Desktop\patent_price\国专局原文本信息\current_all\current_all"
path_list, name_list = read_each_file(filePathC)
name_list_strip = list()
for i in name_list:
    name_list_strip.append(i.split("_")[0].strip("CN"))

content_list = list()
for path in path_list:
    content_list.append(read_file(path))

workbook = xlrd.open_workbook('C:\\Users\skydata\Desktop\patent_price\price_info1.xlsx')
booksheet = workbook.sheet_by_index(0)  # 用索引取第一个sheet
patent_number = booksheet.col_values(0)[:8182]  # 专利号
patent_number_strip = list()
for number in patent_number:
    patent_number_strip.append(number[0:12])
patent_name = booksheet.col_values(1)[:8182]  # 专利名
patent_trade_type = booksheet.col_values(2)[:8182]  # 专利交易类型
patent_trade_date = booksheet.col_values(3)[:8182]  # 专利交易日期
patent_trade_price = booksheet.col_values(4)[:8182]  # 专利交易价格
patent_price_type = booksheet.col_values(5)[:8182]  # 专利价格类型
patent_detail = booksheet.col_values(6)[:8182]  # 备注
patent_price_list = list()
for i in patent_trade_price:
    patent_price_list.append([float(i)])
price_list = list()
for name in name_list_strip:
    if name in patent_number_strip:
        index = patent_number_strip.index(name)
        price_list.append(patent_price_list[index])
    else:
        price_list.append(0)

print(len(path_list), len(name_list), len(content_list), len(price_list))
df = pd.DataFrame()
df.loc[:, 'path_list'] = [i for i in path_list]
df.loc[:, 'name_list'] = [i for i in name_list]
df.loc[:, 'content_list'] = [i for i in content_list]
df.loc[:, 'price_list'] = [i for i in price_list]
df_new = pd.DataFrame()
df_new = df[df['price_list'] != 0]
# print(rm_index_list)
print(df_new)
df_new.to_pickle('2185_data_directory\patent_data_2185.pkl')
