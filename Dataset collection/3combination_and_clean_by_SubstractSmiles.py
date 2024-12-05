#!/usr/bin/python
# coding: utf-8

import csv
import json


#Kcat/Km

with open(r"F:\SecondStudy\sabio_ratio_clean_smiles.json", 'r') as infile:
    sabio_name_smiles = json.load(infile)

with open(r"F:\SecondStudy\brenda_ratio_clean_smiles.json", 'r') as infile:
    brenda_name_smiles = json.load(infile)

with open(r"F:\SecondStudy\brenda_ratio_clean.tsv", "r", encoding='utf-8') as file1 :
    brenda_lines = file1.readlines()[1:]

with open(r"F:\SecondStudy\sabio_ratio_clean.tsv", "r", encoding='utf-8') as file2 :
    sabio_lines = file2.readlines()[1:]

Kcat_data = list()
Kcat_data_include_value = list()
Substrate_name = dict()
Substrate_smiles = dict()
entry_uniprot = dict()

for line in brenda_lines : #根据smiles整理brenda Kcat/km数据
    # print(line)
    data = line.split('\t')
    ECNumber = data[2]
    Substrate = data[3]
    EnzymeType = set(data[4].split('/'))
    Organism =data[5]
    pH = data[6]
    Temp = data[7]
    Value = data[-2]
    Unit = data[-1]

    smiles = brenda_name_smiles[Substrate]
    # print(smiles)
    if smiles is not None :
        # print(smiles)
        Substrate_name[Substrate.lower()] = Substrate
        Substrate_smiles[Substrate.lower()+'&smiles'] = smiles

        Kcat_data_include_value.append([ECNumber, Substrate.lower(), EnzymeType, Organism, pH, Temp, Value, Unit])
        Kcat_data.append([ECNumber, Substrate.lower(), EnzymeType, Organism, pH, Temp])

for line in sabio_lines :  #根据smiles整理Sabio-RK Kcat/km数据
    # print(line)
    data = line.strip().split('\t')
    ECNumber = data[1]
    Substrate = data[2]
    EnzymeType = set(data[3].split('/'))
    Organism =data[5]
    UniprotID = data[6]
    pH = data[7]
    Temp = data[8]
    Value = data[9]
    Unit = data[10]

    smiles = sabio_name_smiles[Substrate]
    # print(smiles)
    if smiles is not None :
        # print(smiles)
        Substrate_name[Substrate.lower()] = Substrate
        Substrate_smiles[Substrate.lower()+'&smiles'] = smiles
        entry_uniprot[ECNumber + Substrate.lower() + Organism] = UniprotID

        Kcat_data_include_value.append([ECNumber, Substrate.lower(), EnzymeType, Organism, pH, Temp, Value, Unit])
        Kcat_data.append([ECNumber, Substrate.lower(), EnzymeType, Organism, pH, Temp])

print(len(Kcat_data))  # 32716 两个数据集合并

# 合并 Brenda 和 Sabio-RK Kcat/km数据，并根据底物名字去重，选最大值
new_lines = list()
for line in Kcat_data :
    if line not in new_lines :
        new_lines.append(line)

print(len(new_lines))  # 32310

i = 0
clean_Kcat = list()
for new_line in new_lines :
    # print(new_line)
    i += 1
    # print(i)
    value_unit = dict()
    Kcat_values = list()
    for line in Kcat_data_include_value :
        if line[:-2] == new_line :
            value = line[-2]
            value_unit[str(float(value))] = line[-1]
            # print(type(value))  # <class 'str'>
            Kcat_values.append(float(value))
    # print(value_unit)
    # print(Kcat_values)
    max_value = max(Kcat_values) # choose the maximum one for duplication Kcat value under the same entry as the data what we use
    unit = value_unit[str(max_value)]
    # print(max_value)
    # print(unit)
    Substrate = Substrate_name[new_line[1]]
    Smiles = Substrate_smiles[new_line[1]+'&smiles']
    try :
        UniprotID = entry_uniprot[new_line[0]+new_line[1]+new_line[3]]
    except :
        UniprotID = ''
    new_line[1] = Substrate
    new_line[2] = '/'.join(new_line[2])
    new_line = new_line + [Smiles, UniprotID, str(max_value), unit]
    # print(new_line)
    # new_line.append(str(max_value))
    # new_line.append(unit)
    if new_line[-1] == 'mM^(-1)*s^(-1)' :
        clean_Kcat.append(new_line)

# print(clean_Kcat)
print(len(clean_Kcat))  # 31369

with open(r"F:\SecondStudy\Kcatkm_combination_073000.tsv", "w") as outfile :
    records = ['ECNumber', 'Substrate', 'EnzymeType', 'Organism', 'pH', 'Temp', 'Smiles', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        outfile.write('\t'.join(line) + '\n')

# The above is to eliminate the dupliaction entries by Substrate name
print('-----------------------------------------------')

## This is to eliminate the duplication entries by Smiles
with open(r"F:\SecondStudy\Kcatkm_combination_073000.tsv", "r", encoding='utf-8') as infile :
    Kcat_lines = infile.readlines()[1:]

Kcat_data = list()
Kcat_data_include_value = list()
Substrate_name = dict()
entry_uniprot = dict()

for line in Kcat_lines :
    # print(line)
    data = line.strip().split('\t')
    ECNumber = data[0]
    Substrate = data[1]
    EnzymeType = set(data[2].split('/'))
    Organism =data[3]
    Smiles = data[6]
    UniprotID = data[7]
    pH = data[4]
    Temp = data[5]
    Value = data[8]
    Unit = data[9]

    Substrate_name[Smiles] = Substrate
    entry_uniprot[ECNumber + Smiles + Organism] = UniprotID

    Kcat_data_include_value.append([ECNumber, EnzymeType, Organism, Smiles, pH, Temp, Value, Unit])
    Kcat_data.append([ECNumber, EnzymeType, Organism, Smiles, pH, Temp])

print(len(Kcat_data))  # 31369


new_lines = list()
for line in Kcat_data :
    if line not in new_lines :
        new_lines.append(line)

print(len(new_lines))  # 29949
i = 0
clean_Kcat = list()
for new_line in new_lines :
    # print(new_line)
    i += 1
    print(i)
    value_unit = dict()
    Kcat_values = list()
    for line in Kcat_data_include_value :
        if line[:-2] == new_line :
            value = line[-2]
            value_unit[str(float(value))] = line[-1]
            # print(type(value))  # <class 'str'>
            Kcat_values.append(float(value))
    # print(value_unit)
    # print(Kcat_values)
    max_value = max(Kcat_values) # choose the maximum one for duplication Kcat value under the same entry as the data what we use
    unit = value_unit[str(max_value)]
    # print(max_value)
    # print(unit)
    Substrate = Substrate_name[new_line[3]]
    try :
        UniprotID = entry_uniprot[new_line[0]+new_line[3]+new_line[2]]
    except :
        UniprotID = ''
    new_line[1] = '/'.join(new_line[1])
    new_line = new_line + [Substrate, UniprotID, str(max_value), unit]
    # print(new_line)
    # new_line.append(str(max_value))
    # new_line.append(unit)
    if new_line[-1] == 'mM^(-1)*s^(-1)' :
        clean_Kcat.append(new_line)

# print(clean_Kcat)
print(len(clean_Kcat))  # 12858

with open(r"F:\SecondStudy\Kcatkm_combination_073022.tsv", "w") as outfile :
    records = ['ECNumber', 'EnzymeType', 'Organism', 'Smiles', 'pH', 'Temp', 'Substrate', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        outfile.write('\t'.join(line) + '\n')

