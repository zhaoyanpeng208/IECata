import json
from typing import List

import pandas as pd
from rdkit import Chem


def smiles2inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToInchiKey(mol)


def preprocess(data):
    data["Type"] = data.Type.map(lambda x: x.title())
    # data["Substrate"] = data["Substrate"].map(lambda x: x.lower())
    data = data[data["Smiles"].notna()].copy()
    data["InChIKey"] = data["Smiles"].map(lambda x: smiles2inchikey(x))
    return data


def keep_max(data: pd.DataFrame):
    _out = data.iloc[0, :].copy()
    _out["Value"] = data["Value"].values.max()
    return _out


if __name__ == "__main__":
    # sabio = pd.read_csv('D:/Downloads/final_data/Kcat_sabio_PhTemp_clean_WM_trycombina.tsv', sep="\t")
    # brenda = pd.read_csv('D:/Downloads/final_data/KCAT_pHTemp/Kcat_brenda_PhTemp_clean.tsv', sep="\t")

    sabio = pd.read_csv('D:/Downloads/final_data/KM_pHTemp/KM_sabio_PhTemp_clean_WM_trycombina.tsv', sep="\t")
    brenda = pd.read_csv('D:/Downloads/final_data/KM_pHTemp/KM_brenda_PhTemp_clean22222.tsv', sep="\t")

    with open("D:/Downloads/final_data/KM_pHTemp/KM_sabio_PHtemp_smiles.json") as sabio_reader:
        sabio_mapping = json.load(sabio_reader)##摆设

    sabio["Smiles"] = sabio.Substrate.map(lambda x: sabio_mapping[x])

    with open("D:/Downloads/final_data/KM_brenda_smiles_mapping.json") as f:
        brenda_mapping = json.load(f) #爬过的

    brenda["Smiles"] = brenda.Substrate.map(lambda x: brenda_mapping.get(x.lower(), None))

    infer_keys = list(brenda.columns[:-3])
    if "EntryID" in infer_keys:
        infer_keys.remove("EntryID")
    # infer_keys.remove("Substrate")
    infer_keys += ["Smiles"]

    used_keys = infer_keys + ["Value", "Unit"]

    tmp = pd.concat([
        sabio.loc[:, used_keys + ["UniprotID"]],
        brenda.loc[:, used_keys]
    ], axis=0)

    tmp = preprocess(tmp)

    # tmp = pd.read_csv("/home/zyp/DLKcat-master/DeeplearningApproach/Code/preprocess/Data/database/KCAT_pHTemp/brenda_data_with_inchikey.csv")
    infer_keys = list(tmp.columns)[:-4] + ["InChIKey"]
    out = tmp.groupby(infer_keys).apply(keep_max)
    out.to_csv('D:/Downloads/final_data/KM_pHTemp/KM_brenda_clean_smiles.tsv', sep="\t", index=False)

    print("DONE!!!")
