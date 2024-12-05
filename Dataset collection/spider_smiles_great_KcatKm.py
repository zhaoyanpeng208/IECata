import json
import urllib

from lxml import etree
import requests
import re

from urllib.parse import quote


class SabioSpider:
    @staticmethod
    def search(name: str) -> str:
        url = "http://sabio.h-its.org/newSearch/search/"
        data = {
            "q": f'  Substrate:"{name}"',
            "ontologySearch": "falsex",
            "wildtype": "true",
            "mutant": "true",
            "recombinant": "false",
            "kineticData": "false",
            "transportReaction": "false",
            "phValues": "0 - 14",
            "temperatures": "-10.0 C° - 115.0 C°",
            "directSubmission": "true",
            "journal": "true",
            "biomodel": "true",
            "date": "false",
            "entryDate": "14/10/2008",
            "ipAddress": "127.0.1.1",
            "ipAddress2": "192.168.56.1",
            "remoteHost": "192.168.56.1",
            "view": "entry",
        }
        resp = requests.post(url, data=data)
        if resp.status_code != 200:
            return ''

        entries = re.findall(' Entry ID: (\d+)', resp.text)
        if len(entries) == 0:
            return ''

        entry = entries[0]
        url = f"http://sabio.h-its.org/kindatadirectiframe.jsp?kinlawid={entry}&newinterface=true"
        resp = requests.get(url)
        if resp.status_code != 200:
            return ''

        tree = etree.HTML(resp.text)
        _click = tree.xpath(f'//td[contains(text(), "{name}")]/@onclick')
        if len(_click) == 0:
            return ''

        flag = re.findall('cid=(\d+)', _click[0])

        if flag:
            return flag[0]
        return ""

    @staticmethod
    def name_to_smiles(name: str) -> str:
        cid = SabioSpider.search(name)
        if cid:
            url = f"http://sabio.h-its.org/compdetails.jsp?cid={cid}"
            resp = requests.get(url)
            if resp.status_code == 200:
                tree = etree.HTML(resp.text)
                flag = tree.xpath('//*[@id="Smiles_0"]/text()')
                if len(flag) > 0:
                    return flag[0]
        return ""


class BrendaSpider:
    @staticmethod
    def search(name: str) -> str:
        escape_name = quote(name)
        url = f"https://www.brenda-enzymes.org/search_result.php?quicksearch=1&noOfResults=10&a=13&W[3]={escape_name}&T[3]=2&V[8]=1"

        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return ""

        tree = etree.HTML(resp.text)
        _cid_href = tree.xpath(f'//a[text()="{name}"]/@href')
        if len(_cid_href) == 0:
            return ""
        cid_href = _cid_href[0]

        new_url = f"https://www.brenda-enzymes.org/{cid_href}"
        resp = requests.get(new_url, timeout=30)
        if resp.status_code != 200:
            return ""

        tree = etree.HTML(resp.text)
        inchikeys = tree.xpath('//*[@id="flatcontent"]/div[2]/div[6]/div[2]/div[3]/text()')
        if len(inchikeys) == 0:
            return ""

        return inchikeys[0]

    @staticmethod
    def name_to_smiles(name: str) -> str:
        inchikey = BrendaSpider.search(name)
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/" \
                      f"property/CanonicalSMILES/JSON/"
        resp = requests.get(pubchem_url)
        if resp.status_code != 200:
            return ""

        data = json.loads(resp.text)
        try:
            props = data["PropertyTable"]["Properties"]
            if len(props) == 0:
                return ""
            return props[0]["CanonicalSMILES"]
        except KeyError:
            return ""


class UniprotSpider:
    @staticmethod
    def retrieve_uniprot(ec, organism):  # 找没有UniportId的EC号与organism
        url = "https://rest.uniprot.org/uniprotkb/search/"
        params = {"query": f"ec:{ec} AND organism_name:{organism} AND reviewed: true", "format": "fasta"}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            flag = re.findall("sp\|(.*?)\|", response.text, re.S)
            if len(flag) > 0:
                return "|".join(flag)

        return ""


if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    #
    mapping = {}
    for file in ["F:/SecondStudy/brenda_ratio_smiles.json",
                 "F:/SecondStudy/sabio_ratio_smiles.json"]:
        with open(file) as f:
            mapping = {**mapping, **json.load(f)}  # 合并两个字典，并且当有相同键时，后一个字典的优先度更高

    mapping = {k.lower(): v for k, v in mapping.items() if v is not None}  # 把映射表中value为None的值舍去

    data = pd.read_csv("F:/SecondStudy/sabio_ratio_clean.tsv")
    substrates = list(set(data["Substrate"]))

    for _substrate in substrates:
        substrate = _substrate.lower()
        if substrate in mapping:
            continue

        for spider in [BrendaSpider.name_to_smiles, SabioSpider.name_to_smiles]:
            try:
                new_smi = spider(substrate)
            except requests.exceptions.ConnectionError:
                with open("F:/SecondStudy/sabio_ratio_error.log", 'a') as f:
                    f.write(substrate)
                    f.write("\n")
                continue
            except requests.exceptions.Timeout:
                with open("F:/SecondStudy/sabio_ratio_error.log", 'a') as f:
                    f.write(substrate)
                    f.write("\n")
                continue

            if new_smi:
                mapping[substrate] = new_smi
                with open("F:/SecondStudy/sabio_ratio_smiles_mapping.json", 'w') as w:
                    json.dump(mapping, w)
                break

    # data = pd.read_csv("./FIN/Kcat_clean_data/Kcat_brenda_PHTemp_clean.csv")
    # pairs = list(set(data.apply(lambda x: f'{x["ECNumber"]};{x["Organism"]}', axis=1)))
    #
    # mapping = {}
    # for _pair in tqdm(pairs):
    #     pair = _pair.split(";")
    #     mapping[_pair] = UniprotSpider.retrieve_uniprot(*pair)


    print("DONE!!!")

