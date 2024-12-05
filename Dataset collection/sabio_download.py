#!/usr/bin/python
# coding: utf-8

import os

import requests

# Extract EC number list from ExPASy, which is a repository of information relative to the nomenclature of enzymes.
def eclist():
    with open(r'F:\SecondStudy\enzyme.dat', 'r') as outfile :
        lines = outfile.readlines()

    files = os.listdir(r'F:\SecondStudy\sabio_ratio')
    exist_ec = [os.path.splitext(file)[0] for file in files]

    ec_list = list()
    for line in lines :
        if line.startswith('ID') :
            ec = line.strip().split('  ')[1]
            ec_list.append(ec)
    # print(ec_list)
    print(len(ec_list),flush=True) # 7906

    undownloaded_ec = list(set(ec_list) - set(exist_ec))
    return ec_list, undownloaded_ec

def sabio_info(allEC):
    QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv'

    # specify search fields and search terms

    # query_dict = {"Organism":'"lactococcus lactis subsp. lactis bv. diacetylactis"', "Product":'"Tyrosine"'}
    # query_dict = {"Organism":'"lactococcus lactis subsp. lactis bv. diacetylactis"',} #saccharomyces cerevisiae  escherichia coli
    # query_dict = {"ECNumber":'"1.1.1.1"',}
    i = 0
    for EC in allEC :
        # if EC.strip()[0] != "7":
        #     continue
        i += 1
        print('This is %d ----------------------------' %i,flush=True)
        print(EC,flush=True)
        query_dict = {"ECNumber":'%s' %EC,}
        query_string = ' AND '.join(['%s:%s' % (k,v) for k,v in query_dict.items()])


        # specify output fields and send request


        query = {'fields[]': ['EntryID', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'ECNumber', 'Parameter', 'temperature', 'pH'], 'q': query_string}
        # query = {'fields[]': ['EntryID', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'ECNumber','Parameter'],'q': query_string}
        # the 'Smiles' keyword could get all the smiles included in substrate and product

        request = requests.post(QUERY_URL, params = query, verify=False)
        # request.raise_for_status()


        # results
        results = request.text
        print(results,flush=True)
        print('---------------------------------------------')

        if results :
            with open(r'F:\SecondStudy\sabio_ratio\%s.txt' %EC, 'w', encoding='utf=8') as ECfile :
                ECfile.write(results)


if __name__ == '__main__' :
    allEC, un_ec = eclist()
    sabio_info(un_ec)#
    # sabio_info(allEC)


