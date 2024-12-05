#!/usr/bin/python
# coding: utf-8
# -- coding: utf-8 --**

import os
import re
import json
import requests
# from urllib import request
import time
import urllib3.request
# import urllib2 ## 用于第一个函数
from zeep import Client
import hashlib
import io
# import string
# import hashlib
# from SOAPpy import WSDL
from SOAPpy import SOAPProxy ## for usage without WSDL file


# This function is to obtain the protein sequence according to the protein id from Uniprot API
# # # # # # https://www.uniprot.org/uniprot/A0A1D8PIP5.fasta
# https://www.uniprot.org/help/api_idmapping
'''
def uniprot_sequence(id) :
    url = "https://www.uniprot.org/uniprot/%s.fasta" % id
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(id, "can not find from uniprot!")
            return
        respdata = resp.text.strip()
        seq = "".join(respdata.split("\n")[1:])
        print(seq)
        return seq
    except:
        print(id, "can not find from uniprot!")
        return None

# if __name__ == '__uniprot_sequence__':
#     uniprot_sequence()
    
def uniprotID_entry() :
    # 找到所有有uniportID的序列
    # uniprot_sequence('P18314')
    with open(r'F:\SecondStudy\Kotori\6combination_clean.tsv', "r") as file :
    # with open(r"F:\SecondStudy\try_uniprotID.csv", "r") as file :

        combination_lines = file.readlines()[1:]

    uniprotID_list = list()
    uniprotID_seq = dict()
    uniprotID_noseq = list()

    i=0
    for line in combination_lines :
        data = line.strip().split('\t')
        uniprotID = data[4]

        if uniprotID :
        #     seq = uniprot_sequence('P49384')
            if ' ' in uniprotID :
                # i += 1  # 561
                # print(i)
                # print(uniprotID.split(' '))
                uniprotID_list += uniprotID.split(' ')
            else :
                # print(uniprotID)
                uniprotID_list.append(uniprotID)

    print(len(uniprotID_list))    # 13164
    uniprotID_unique = list(set(uniprotID_list))
    print(len(uniprotID_unique)) # 1685
    print(uniprotID_unique[-6:])  ###？？？？？？['O25613', 'Q1QV19', 'P21266', 'P38998', 'P13922', 'P37351']

    for uniprotID in uniprotID_unique :
        i += 1
        print(i)
        sequence = uniprot_sequence(uniprotID)
        if sequence :
            uniprotID_seq[uniprotID] = sequence
        else :
            uniprotID_noseq.append(uniprotID)

    print(len(uniprotID_seq))  # 1598
    print(len(uniprotID_noseq))  # 17
    print(uniprotID_noseq)

    # ['P0A5R0', 'P0C5C1', 'Q02469', 'P96420', 'Q9N1E2', 'P15651', 'P15650', 'A0A024BTN9', 'O05783', 'P0A4X4', 'D4ZTT4', 'P51698', 'O60344', 'P56967', 'P0A4Z2', 'Q47741', 'A4VVM9']
    # check one by one
    # ['B5HSR1',  ,
    # 'P63454',

    with open(r'F:\SecondStudy\Kotori\combination_clean_uniprotID_entry.json', 'w') as outfile :
        json.dump(uniprotID_seq, outfile, indent=4)

def uniprotID_noseq() :
#找UniportID改变的那些ID的序列
    with open(r"F:\SecondStudy\Kcatkm_uniprotID_entry.json", 'r') as infile :
        uniprotID_seq = json.load(infile)

    print(len(uniprotID_seq)) # 1958

     # ['P0A5R0', 'O60344', 'P56967', 'D4ZTT4', 'Q47741', 'P96420', 'A4VVM9', 'Q02469', 'P0A4Z2', 'O05783', 'P0C5C1', 'P0A4X4', 'A0A0D1LMH2', 'P51698']
    uniprotID_noseq = {'P0A5R0': 'P9WIL4', 'P0C5C1': 'P9WKD2', 'Q02469': 'P0C278', 'P96420': 'P9WQB2', 'O05783': 'P9WIQ2', 'P0A4X4': 'P9WQ86',
                       'P51698': 'A0A1L5BTC1', 'O60344': 'P0DPD6', 'P56967': 'F2MMP0', 'P0A4Z2': 'P9WPY2', 'Q47741': 'F2MMN9',
                       'Q9N1E2': 'Q9N1E2', 'P15651': 'P15651', 'P15650': 'P15650', 'A0A024BTN9': 'A0A024BTN9', 'D4ZTT4': 'D4ZTT4', 'A4VVM9': 'A4VVM9'}
    # 最后一列ID都没变，第二次D4ZTT4 A4VVM9仍没找到序列，手动添加的。

    for uniprotID, mappedID in uniprotID_noseq.items() :
        sequence = uniprot_sequence(mappedID)
        print(uniprotID)
        print(sequence)
        if sequence :
            uniprotID_seq[uniprotID] = sequence   #？？？？？？？？？？？？？？？？？？？？？？？？
        else :
            print('No sequence found!---------------------------')

    print(len(uniprotID_seq))  # 1613 #第二次D4ZTT4 A4VVM9仍没找到序列，需手动添加的。所以最终长度应是1615

    with open(r"F:\SecondStudy\Kcatkm_uniprotID_entry_all.json", 'w') as outfile :
        json.dump(uniprotID_seq, outfile, indent=4)

# You can try to retrieve sequences from uniprot using rest interface.
# Example: (ec: 1.1.1.1 , organisms: Homo sapiens)
# http://www.uniprot.org/uniprot/?query=ec:1.1.1.1+AND+organism:"Homo sapiens"&format=fasta
# full information abut syntax you can find here: http://www.uniprot.org/help/programmatic_access
def seq_by_ec_organism(ec, organism) :
    IdSeq = dict()
    # https://www.biostars.org/p/356687/
    params = {"query": "ec:%s AND organism:%s AND reviewed:yes" % (ec, organism), "format": "fasta"}
    response = requests.get("http://www.uniprot.org/uniprot/", params=params)
    print(type(response.text)) # <class 'str'>

    try :
        # respdata = response.text.strip()
        # # print(respdata)
        # IdSeq[ec+'&'+organism] =  "".join(respdata.split("\n")[1:])

        respdata = response.text
        print(respdata)
        sequence = list()
        seq = dict()
        i = 0
        for line in respdata.split('\n') :
            if line.startswith('>') :
                name=line
                seq[name] = ''
            else :
                seq[name] += line.replace('\n', '').strip()
        IdSeq[ec+'&'+organism] =  list(seq.values())

    except :
        print(ec+'&'+organism, "can not find from uniprot!")
        IdSeq[ec+'&'+organism] = None

    print(IdSeq[ec+'&'+organism])
    return IdSeq[ec+'&'+organism]
'''

'''
# Run in python 2.7
#bug_list用于存储第一次爬虫导致中断的ec+organism，可循环跑bug_list，直至找不到序列为止
bug_list_Kcatkm = []
f = open('F:/SecondStudy/Kotori/bug_list_Kcatkm.tsv', 'w')
def seq_by_brenda(ec, organism):
    # E-mail in BRENDA:
    email = 'leyu@chalmers.se'
    # Password in BRENDA:
    password = 'yuanle13579'

    endpointURL = "https://www.brenda-enzymes.org/soap/brenda_server.php"
    client      = SOAPProxy(endpointURL)
    password    = hashlib.sha256(password).hexdigest()
    credentials = email + ',' + password

    parameters = credentials+","+"ecNumber*%s#organism*%s" %(ec, organism)
    try:
        content = client.getSequence(parameters)
        # E-mail in BRENDA:
        # email = 'leyu@chalmers.se'
        # # Password in BRENDA:
        # password = 'yuanle13579'
        # wsdl = "https://www.brenda-enzymes.org/soap/brenda.wsdl"
        # client      = WSDL.Proxy(wsdl)
        # password    = hashlib.sha256(password).hexdigest()
        # credentials = email + ',' + password
        #
        # parameters = credentials+","+"ecNumber*%s#organism*%s" %(ec, organism)
        # content = client.getSequence(parameters)
        split_sequences = content.strip().split('!') #noOfAminoAcids #!
        # UniProtKB/TrEMBL is a computer-annotated protein sequence database complementing the UniProtKB/Swiss-Prot Protein Knowledgebase.
        sequences = list()
        # print(split_sequences)
        for sequence in split_sequences :
            dict_entry = dict()
            # print(sequence)
            list_one = sequence.split('#')
            # print(list_one)
            for one in list_one[:-1] :
                # print(one)
                dict_entry[one.split('*')[0]] = one.split('*')[1]
            # try :
            #     if dict_entry['source'] == 'Swiss-Prot' :
            #         sequences.append(dict_entry['sequence'])
            #     else :
            #         continue
            # except :
            #     sequences = None
            try :
                sequences.append(dict_entry['sequence'])
            except :
                sequences = None

        print(sequences)   ###673
        return sequences

    except:
        bug_list_Kcatkm.append("%s,%s" %(ec, organism))
        f.write("%s,%s\n" %(ec, organism))
        print ("%s,%s" %(ec, organism))

'''
def nouniprotID_entry_uniprot() :
    # ec = '1.1.1.206'
    # organism = 'Datura stramonium'
    # seq_by_ec_organism(ec, organism)


    with open("/home/zyp/DLKcat-master/DeeplearningApproach/Code/preprocess/Data/database/Kcatkm/Kcatkm_combination_0731.tsv", "r") as file :
        combination_lines = file.readlines()[1:]

    IdSeq = dict()
    entries = list()
    i=0
    for line in combination_lines :
        data = line.strip().split('\t')
        ec = data[0]
        organism = data[2]
        uniprotID = data[5]

        if not uniprotID :
            entries.append((ec,organism))

    print(len(entries))  # 673 需要通过EC和organism找蛋白序列的entries
    entries_unique = set(entries)
    print(len(entries_unique)) # 176

    for entry in list(entries_unique) :
        # print(entry)
        ec, organism = entry[0], entry[1]
        i += 1
        # if i<10:
        print('This is', str(i)+'------------')
        # else:
        #     break
        IdSeq[ec+'&'+organism] = seq_by_ec_organism(ec, organism)

    # print(len(IdSeq)
        if i%10 == 0 :
            time.sleep(3)

    with open('/home/zyp/DLKcat-master/DeeplearningApproach/Code/preprocess/Data/database/Kcatkm/Kcatkm_nouniprotID_entry_all.json', 'w') as outfile :
        json.dump(IdSeq, outfile, indent=4)


def combine_sequence() :
    with open(r"F:\SecondStudy\Kotori\6combination_uniprotID_entry.json", 'r') as file1:
        uniprot_file1 = json.load(file1)

    # with open('/home/zyp/DLKcat-master/DeeplearningApproach/Code/preprocess/Data/database/Kcatkm/nouniprotID_entry_all.json', 'r') as file2:  # By Uniprot API
    #     nouniprot_file2 = json.load(file2)

    with open(r"F:\SecondStudy\Kotori\nouniprotID_entry_brenda.json", 'r') as file3:  # By BRENDA API
        nouniprot_file3 = json.load(file3)
        # DATA = []
        # for line in file3.readlines():
        #     nouniprot_file3 = json.loads(line)
        #     DATA.append(nouniprot_file3)

#  Kcatkm_nouniprotID_entry_brenda.json 太大，只能一行一行的读，不能直接用load直接加载整个json文件
# loads() 传的是json字符串，而 load() 传的是文件对象
# 使用 loads() 时需要先读取文件在使用，而 load() 则不用
    with open(r"F:\SecondStudy\Kotori\combination_clean_by_smiles.tsv", "r") as file4 :
        Kcat_lines = file4.readlines()[1:]

    # i = 0
    # for proteinKey, sequence in nouniprot_file2.items() :
    #     if sequence :
    #         if len(sequence) == 1 :  # 1178 BRENDA  1919 Uniprot
    #         # if sequence :  # 1784 BRENDA  3363 Uniprot
    #             i += 1   
    #             print(i)
    # print(len(nouniprot_file3))

    i = 0
    j = 0
    n = 0
    entries = list()
    for line in Kcat_lines :
        data = line.strip().split('\t')
        ECNumber, EnzymeType, Organism, Smiles = data[0], data[1], data[2], data[3]
        pH, Temp, Substrate, UniprotID, Value, Unit = data[4], data[5], data[6], data[7], data[8], data[9]

        RetrievedSeq = ''
        entry = dict()
        # print(UniprotID)
        if UniprotID :
            # print(UniprotID)
            try :  # because a few (maybe four) UniprotIDs have no ID as the key 
                if ' ' not in UniprotID :
                    RetrievedSeq = [uniprot_file1[UniprotID]]
                    # print(RetrievedSeq)
                else :
                    # print(UniprotID)
                    RetrievedSeq1 = [uniprot_file1[UniprotID.split(' ')[0]]] ##.split(' ')[0]以空格划分的第一部分
                    RetrievedSeq2 = [uniprot_file1[UniprotID.split(' ')[1]]]
                    if RetrievedSeq1 == RetrievedSeq2 :
                        RetrievedSeq = RetrievedSeq1
                    # if len(RetrievedSeq) == 1:
                    #     print(RetrievedSeq)
            except :
                continue

        else :
            if nouniprot_file3[ECNumber+'&'+Organism] :
                # print(nouniprot_file3[ECNumber+'&'+Organism])
                if len(nouniprot_file3[ECNumber+'&'+Organism]) == 1 :
                    RetrievedSeq = nouniprot_file3[ECNumber+'&'+Organism]
                    # print(RetrievedSeq)
                else :
                    RetrievedSeq = ''

        # print(RetrievedSeq)
        try:  # local variable 'RetrievedSeq' referenced before assignment
            if len(RetrievedSeq) == 1 and 'wildtype' in EnzymeType:  # 21108 for all, 9529 wildtype, 11579 mutant (EnzymeType != 'wildtype')
                sequence = RetrievedSeq
                i += 1
                # print(str(i) + '---------------------------')
                # print(ECNumber)
                # print(Organism)
                # print(sequence)

                entry = {
                    'ECNumber': ECNumber,
                    'Organism': Organism,
                    'Smiles': Smiles,
                    'Substrate': Substrate,
                    'Sequence': sequence[0],
                    'Type': 'wildtype',
                    'Value': Value,
                    'Unit': Unit,
                }

                entries.append(entry)

            if len(RetrievedSeq) == 1 and EnzymeType != 'wildtype':
                sequence = RetrievedSeq[0]

                mutantSites = EnzymeType.split('/')
                # print(mutantSites)

                mutant1_1 = [mutantSite[1:-1] for mutantSite in mutantSites]
                mutant1_2 = [mutantSite for mutantSite in mutantSites]
                mutant1 = [mutant1_1, mutant1_2]
                mutant2 = set(mutant1[0])
                if len(mutant1[0]) != len(mutant2) :
                    print(mutant1)
                    n += 1
                    print(str(n) + '---------------------------')  # some are mapped, some are not mapped. R234G/R234K (60, 43 mapped, 17 not mapped)

                mutatedSeq = sequence
                for mutantSite in mutantSites :
                    # print(mutantSite)
                    # print(mutatedSeq[int(mutantSite[1:-1])-1])
                    # print(mutantSite[0])
                    # print(mutantSite[-1])
                    if mutatedSeq[int(mutantSite[1:-1])-1] == mutantSite[0] :
                        # pass
                        mutatedSeq = list(mutatedSeq)
                        mutatedSeq[int(mutantSite[1:-1])-1] = mutantSite[-1]
                        mutatedSeq = ''.join(mutatedSeq)
                        if not mutatedSeq :
                            print('-------------')
                    else :
                        # n += 1
                        # print(str(n) + '---------------------------')
                        mutatedSeq = ''

                if mutatedSeq :
                    # j += 1
                    # print(str(j) + '---------------------------')          
                    entry = {
                        'ECNumber': ECNumber,
                        'Organism': Organism,
                        'Smiles': Smiles,
                        'Substrate': Substrate,
                        'Sequence': mutatedSeq,
                        'Type': 'mutant',
                        'Value': Value,
                        'Unit': Unit,
                    }

                    entries.append(entry)

        except:
            continue

    # mutatedSeq.replace([int(mutantSite[1:-1])-1], mutantSite[-1])
    print(i)      #7079  野生型数量

    print(len(entries))   # 12054  总数量

    with open(r"F:\SecondStudy\Kotori\combination_fina2.tsv", 'w') as outfile :
        json.dump(entries, outfile, indent=4)
    # with open(r'F:\SecondStudy\Kcatkm_combination_0918_wildtype_mutant.json', 'w') as outfile :
    #     json.dump(entries, outfile, indent=4)

def check_substrate_seq() :
    with open('/home/zyp/DLKcat-master/DeeplearningApproach/Code/preprocess/Data/database/Kcatkm/Kcatkm_combination_0918.json', 'r') as file :
        datasets = json.load(file)

    substrate = [data['Substrate'].lower() for data in datasets]
    sequence = [data['Sequence'] for data in datasets]
    organism = [data['Organism'].lower() for data in datasets]
    EC_number = [data['ECNumber'] for data in datasets]

    unique_substrate = len(set(substrate))
    unique_sequence = len(set(sequence))
    unique_organism = len(set(organism))
    unique_EC_number = len(set(EC_number))

    print('The number of unique substrate:', unique_substrate)
    print('The number of unique sequence:', unique_sequence)
    print('The number of unique organism:', unique_organism)
    print('The number of unique EC Number:', unique_EC_number)

# The number of unique substrate: 2136
# The number of unique sequence: 5438
# The number of unique organism: 919
# The number of unique EC Number: 1307

###全部用的py2环境
if __name__ == "__main__" :
    # uniprotID_entry()
    # uniprotID_noseq()
    # nouniprotID_entry_uniprot()
    # nouniprotID_entry_brenda()
    combine_sequence()
    # check_substrate_seq()


