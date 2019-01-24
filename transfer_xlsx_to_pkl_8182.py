# -*- coding: UTF-8 -*-
import xlrd
import pandas as pd
'''
这个文件将excel表的8000多条专利和其价格、类型信息转换成一个dataframe，并持久化存储为pkl文件
'''
workbook = xlrd.open_workbook('C:\\Users\skydata\Desktop\patent_price\price_info1.xlsx')
# 用索引取第一个sheet
booksheet = workbook.sheet_by_index(0)
# 去sheet里面的column
patent_number = booksheet.col_values(0)[:8182]  # 专利号
patent_name = booksheet.col_values(1)[:8182]  # 专利名
patent_trade_type = booksheet.col_values(2)[:8182]  # 专利交易类型
patent_trade_date = booksheet.col_values(3)[:8182]  # 专利交易日期
patent_trade_price = booksheet.col_values(4)[:8182]  # 专利交易价格
patent_price_type = booksheet.col_values(5)[:8182]  # 专利价格类型
patent_detail = booksheet.col_values(6)[:8182]  # 备注
price_list = list()
for i in patent_trade_price:
    price_list.append([float(i)])
data = pd.DataFrame()
data.loc[:, 'patent_number'] = [i for i in patent_number]
data.loc[:, 'patent_name'] = [i for i in patent_name]
data.loc[:, 'patent_trade_type'] = [i for i in patent_trade_type]
data.loc[:, 'patent_trade_date'] = [i for i in patent_trade_date]
data.loc[:, 'patent_trade_price'] = [i for i in price_list]
data.loc[:, 'patent_price_type'] = [i for i in patent_price_type]
data.loc[:, 'patent_detail'] = [i for i in patent_detail]
data.to_pickle('data_directory\patent_data_sorted_8182.pkl')
