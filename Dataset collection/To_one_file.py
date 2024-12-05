#!/usr/bin/python
# coding: utf-8

import os
import csv
import re

# outfile = open("F:/SecondStudy/sabio_ratio.tsv", "wt")
outfile = open("F:/SecondStudy/brenda_ratio.tsv", "wt")
tsv_writer = csv.writer(outfile, delimiter="\t")
tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", 'EnzymeType', "Organism", "pH", "Temp","Value", "Unit"])

root = "F:/SecondStudy/sabio_ratio"
filenames = os.listdir(root)
# print(len(filenames)) # 3339 EC files

i = 0
j = 0
for filename in filenames:
    # print(filename[2:-4])
    if filename != '.DS_Store':
        path = os.path.join(root, filename)
        with open(path) as f:
            lines = f.readlines()

    # for line in lines[1:]:
    for line in lines:
        data = line.strip().split('\t')
        value = float(data[2])
        desc = data[5]
        if value > 0:
            i += 1
            if 'mutant' in desc or 'mutated' in desc:
                mutant = re.findall('[A-Z]\d+[A-Z]', desc)  # re is of great use
                # print(mutant)
                if len(mutant) >=1 :
                    enzymeType = '/'.join(mutant)
            else:
                enzymeType = 'wildtype'

            if 'pH' in desc:
                pH = re.findall('pH (.*?)[,|\s]', desc)
                if len(pH) > 0:
                    pH = pH[0]
                else:
                    pH = re.findall('pH (.*)$', desc)
                    if len(pH) > 0:
                        pH = pH[0]
                    else:
                        pH = ''
            else:
                pH = ''

            if pH == "and" or pH == "not" or "%" in pH or ")" in pH:
                pH = ''

            if '&Acirc;&deg;C' in desc:
                temp = re.findall('(\S*)&Acirc;&deg;C', desc)
                if len(temp) > 0:
                    temp = temp[0].replace("*", '')
                else:
                    temp = "*"
            else:
                if '&deg;C' in desc:
                    temp = re.findall('(\S*)&deg;C', desc)
                    if len(temp) > 0:
                        temp = temp[0].replace("*", '')
                    else:
                        temp = "*"
                else:
                    temp = "*"

            temp = temp.split(",")[-1]
            temp = temp.replace(",", "").replace("(", "").replace(")", "")
            pH = "*" if not pH else pH
            temp = "*" if not temp else temp
            # tsv_writer.writerow([i, 'Kcat', filename[2:-4], data[3], enzymeType, data[1], pH, temp, str(value), 's^(-1)'])
            tsv_writer.writerow([i, 'Kcat_KM', filename[2:-4], data[4], enzymeType, data[1], pH, temp,str(value), 'mM^(-1)*s^(-1)'])
            # tsv_writer.writerow([i, 'KM', filename[2:-4], data[3], enzymeType, data[1], pH, temp,str(value), 'mM'])  ### M = mol/L
outfile.close()