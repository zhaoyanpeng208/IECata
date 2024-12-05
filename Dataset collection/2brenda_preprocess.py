#!/usr/bin/python
# coding: utf-8



import re
import os
import csv
import pandas as pd
import numpy as np
from collections import Counter

def convert_value_to_float(string):
    try:
        return float(string)
    except ValueError:
        return np.nan


def foobar(x):
    try:
        if len(x) != 0:
            try:
                tmp = x[0]
                if tmp.endswith(".") or tmp.endswith(","):
                    tmp = tmp[:-1]
                return float(tmp)
            except:
                return np.nan
        return np.nan
    except:
        return np.nan
# ####合并文件夹下所有文件
# root = r"F:\SecondStudy\Kotori\kotori_spider"
#
# # 获取root目录下的所有文件
# files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.txt')]
#
# # 读取所有文件数据并合并
# data = pd.concat([pd.read_csv(file, sep="\t") for file in files])
#
# # 将合并后的数据保存到新的txt文件
# data.to_csv(r"F:\SecondStudy\Kotori\combined_data.txt", sep='\t', index=False)

###去除Values为空的数据
# combine_data = pd.read_csv(r'F:\SecondStudy\Kotori\combined_data.txt', sep='\t')
# combine_data['Values'] = combine_data['Values'].map(convert_value_to_float)
# combine_data = combine_data[combine_data['Values'].notna()]
# combine_data.to_csv(r"F:\SecondStudy\Kotori\combined_data_clean.txt", sep='\t', index=False)

# ###处理commentary列的pH和温度
# data_clean = pd.read_csv(r"F:\SecondStudy\Kotori\combined_data_clean.txt", sep='\t')
# desc = data_clean['Commentary']
#
# # data_clean = data_clean[data_clean.Values > 0].copy()
#
# flag = data_clean['Commentary'].str.contains('mutant', case=False) | data_clean['Commentary'].str.contains('mutated', case=False)
# mutated_log = []
# pH = []
# Temp = []
# # enzymeType = []
#
# for i, description in enumerate(desc):
#     if flag[i]:
#         tmp = re.findall('[A-Z]\d+[A-Z]', description)
#         mutated_log.append('/'.join(tmp))
#     else:
#         mutated_log.append('wildtype')
#
#     try:
#         ph_tmp = re.findall(r"pH\s*(\S*?)[,\s\)]", description)
#         if ph_tmp and ph_tmp[0] != '':
#             if ph_tmp[0].endswith("."):
#                 pH.append(float(ph_tmp[0][:-1]))
#             else:
#                 pH.append(float(ph_tmp[0]))
#         else:
#             ph_tmp = re.findall(r"pH\s*(\S*?)$", description)
#             if ph_tmp and ph_tmp[0] != '':
#                 pH.append(float(ph_tmp[0]))
#             else:
#                 pH.append("*")
#     except:
#         pH.append("*")
#
#     try:
#         temp_tmp = re.findall(r"(\S*?)\s*°[cC]", description)
#         if temp_tmp and temp_tmp[0] != '':
#             if temp_tmp[0].endswith("Â"):
#                 Temp.append(float(temp_tmp[0][:-1]))
#                 continue
#             Temp.append(float(temp_tmp[0]))
#             continue
#         Temp.append("*")
#     except:
#         Temp.append("*")
#
#     # print(mutated_log
#
# data_clean["pH"] = pH
# data_clean["Temp"] = Temp
# data_clean["enzymeType"] = mutated_log
# data_clean['Unit'] = 'mM^(-1)*s^(-1)'
#
# data_clean.to_csv(r"F:\SecondStudy\Kotori\combined_data_clean_fine.txt", sep='\t', index=False)
#
# print("DONE!!!")

#####去重，去pH或Temp缺失的数据

def keep_max(data: pd.DataFrame):
    if data["Values"].min() == 0 or data["Values"].max() / data["Values"].min() > 100:
        return None
    _out = data.iloc[0, :].copy()
    _out["Values"] = data["Values"].values.max()
    return _out

def group_unduplicating(data: pd.DataFrame, infer_columns, func):
    return data.groupby(infer_columns, group_keys=False).apply(func)

def kcat_Km_clean(file: str) -> pd.DataFrame:
    name, ext = os.path.splitext(file)
    sep = "," if ext == ".csv" else "\t"

    try:
        data = pd.read_csv(file, sep=sep)
    except pd.errors.EmptyDataError:
        data = pd.read_csv(file, sep="," if sep == "\t" else ",")

    # 修正Kcat单位
    data = data[data.Unit.isin(["mM^(-1)*s^(-1)"])].copy()
    data["Unit"] = "mM^(-1)*s^(-1)"


    columns = list(data.columns)
    exclude_columns = ["Commentary", "Value", "Unit"]
    infer_columns = [col for col in columns if col not in exclude_columns]
    out = group_unduplicating(data, infer_columns, keep_max)

    # 去掉pH和温度为空的部分，这里使用了isin函数，使得函数更加简介， p.s. '~'表示取反
    nonan = out[(~out.pH.isin(["*", "-"])) & (~out.Temp.isin(["*", "-"]))]
    nonan = nonan.dropna(axis=0, how="all")
    return nonan

if __name__ == "__main__":

    _out = kcat_Km_clean(r"F:\SecondStudy\Kotori\3brenda_data_preprocess_Ph_Temp.txt")
    _out.to_csv(r"F:\SecondStudy\Kotori\4brenda_data_clean_fin.tsv", index=False, sep="\t")