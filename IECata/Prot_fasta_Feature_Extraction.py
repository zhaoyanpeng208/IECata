"""
获取蛋白质预训练序列特征，只需要序列文件即可
生成的文件以字典的形式存储
"""
import torch

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer

import re
import gc
import os
import pandas as pd
import numpy as np
import requests
from tqdm.auto import tqdm

model_name = "Rostlab/prot_t5_xl_uniref50"
# @param {type:"string"}["Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xxl_uniref50", "Rostlab/prot_t5_xxl_bfd", "Rostlab/prot_bert_bfd", "Rostlab/prot_bert", "Rostlab/prot_xlnet", "Rostlab/prot_albert"]

if "t5" in model_name:
    # tokenizer = T5Tokenizer.from_pretrained('/fs1/home/wangll/software/kcatBAN/pretrained/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/ProtT5', do_lower_case=False )
    # model = T5EncoderModel.from_pretrained('/fs1/home/wangll/software/kcatBAN/pretrained/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/ProtT5')
    tokenizer = T5Tokenizer.from_pretrained(
        'D:\zyp\hub\models--Rostlab--prot_t5_xl_uniref50\snapshots\973be27c52ee6474de9c945952a8008aeb2a1a73',
        do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        'D:\zyp\hub\models--Rostlab--prot_t5_xl_uniref50\snapshots\973be27c52ee6474de9c945952a8008aeb2a1a73')
elif "albert" in model_name:
    tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
    model = AlbertModel.from_pretrained(model_name)
elif "bert" in model_name:
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
    model = BertModel.from_pretrained(model_name)
elif "xlnet" in model_name:
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
    model = XLNetModel.from_pretrained(model_name)
else:
    print("Unkown model name")

gc.collect()
print("Number of model parameters is: " + str(int(sum(p.numel() for p in model.parameters())/1000000)) + " Million")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
if torch.cuda.is_available():
    model = model.half()

# 数据集下载，不需要
def downloadNetsurfpDataset():
    netsurfpDatasetTrainUrl = 'https://www.dropbox.com/s/98hovta9qjmmiby/Train_HHblits.csv?dl=1'
    casp12DatasetValidUrl = 'https://www.dropbox.com/s/te0vn0t7ocdkra7/CASP12_HHblits.csv?dl=1'
    cb513DatasetValidUrl = 'https://www.dropbox.com/s/9mat2fqqkcvdr67/CB513_HHblits.csv?dl=1'
    ts115DatasetValidUrl = 'https://www.dropbox.com/s/68pknljl9la8ax3/TS115_HHblits.csv?dl=1'

    datasetFolderPath = "dataset/"
    trainFilePath = os.path.join(datasetFolderPath, 'Train_HHblits.csv')
    casp12testFilePath = os.path.join(datasetFolderPath, 'CASP12_HHblits.csv')
    cb513testFilePath = os.path.join(datasetFolderPath, 'CB513_HHblits.csv')
    ts115testFilePath = os.path.join(datasetFolderPath, 'TS115_HHblits.csv')
    combinedtestFilePath = os.path.join(datasetFolderPath, 'Validation_HHblits.csv')

    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    if not os.path.exists(trainFilePath):
        download_file(netsurfpDatasetTrainUrl, trainFilePath)

    if not os.path.exists(casp12testFilePath):
        download_file(casp12DatasetValidUrl, casp12testFilePath)

    if not os.path.exists(cb513testFilePath):
        download_file(cb513DatasetValidUrl, cb513testFilePath)

    if not os.path.exists(ts115testFilePath):
        download_file(ts115DatasetValidUrl, ts115testFilePath)

    if not os.path.exists(combinedtestFilePath):
        # combine all test dataset files
        combined_csv = pd.concat(
            [pd.read_csv(f) for f in [casp12testFilePath, cb513testFilePath, ts115testFilePath]])
        # export to csv
        combined_csv.to_csv(os.path.join(datasetFolderPath, "Validation_HHblits.csv"),
                            index=False,
                            encoding='utf-8-sig')

# downloadNetsurfpDataset()

#读取数据
def load_dataset(path):
    df = pd.read_csv(path, names=['id', 'input'])
    ids = [id for id in df['id']]
    df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
    df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
    seqs = [list(seq) for seq in df['input_fixed']]

    # df['label_fixed'] = ["".join(label.split()) for label in df['dssp3']]
    # labels = [list(label) for label in df['label_fixed']]
    #
    # df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df['disorder']]
    # disorder = [disorder.split() for disorder in df['disorder_fixed']]
    #
    # assert len(seqs) == len(labels) == len(disorder)
    return ids, seqs

# DTI_ids, DTI_seqs = load_dataset('dataset/bisai/target_info.csv')
# print(DTI_seqs[0][10:30], sep='\n')


def embed_dataset(dataset_seqs, shift_left = 0, shift_right = -1):
    # df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
    # df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
    # seqs = [seq for seq in df['input_fixed']]
    inputs_embedding = []
    seqs = "".join(dataset_seqs.split())
    seqs = re.sub(r"[UZOB]", "X", seqs)
    seqs = [seq for seq in seqs]
    # for sample in tqdm(dataset_seqs):
    with torch.no_grad():
        ids = tokenizer.batch_encode_plus([seqs], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        embedding = model(input_ids=ids['input_ids'].to(device))[0]
        inputs_embedding.append(embedding[0].detach().cpu().numpy()[shift_left:shift_right])

    return embedding[0].detach().cpu().numpy()[shift_left:shift_right]


# Remove any special tokens after embedding
if "t5" in model_name:
    shift_left = 0
    shift_right = -1
elif "bert" in model_name:
    shift_left = 1
    shift_right = -1
elif "xlnet" in model_name:
    shift_left = 0
    shift_right = -2
elif "albert" in model_name:
    shift_left = 1
    shift_right = -1
else:
    print("Unkown model name")

# DTI_seqs_embd = embed_dataset(DTI_seqs, shift_left, shift_right)

# # Example for an embedding output
# print_idx = 0
#
# print("Original Fasta Sequence : ")
# print("".join(DTI_seqs[print_idx]))
#
# # print("Original Sequence labels : ")
# # print("".join(ts115_test_labels[print_idx]))
#
# print("Generated Sequence Features : ")
# print(DTI_seqs_embd[print_idx])
#
# DTI_seqs_dic = {}
# for i in range(len(DTI_seqs)):
#     DTI_seqs_dic[DTI_ids[i]] = DTI_seqs_embd[i]
#
# np.save('./dataset/bisai/protein_emb.npy', DTI_seqs_dic)
