from models import DrugBAN
from time import time
from utils import set_seed, prottrans_graph_collate_func,integer_graph_collate_func, mkdir, \
    Smooth_Label_CBW, Smooth_Label_CSW, Smooth_Label_DMW, Smooth_Label_LDS
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="IECata for kcat prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    # 从config.py读取config
    cfg = get_cfg_defaults()
    # 与yaml文件合并
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    # dataFolder = os.path.join(dataFolder, str(args.split))

    # read data
    data_path = os.path.join(dataFolder, 'train/Kcatkm_total.csv')
    df_data = pd.read_csv(data_path)

    df_train = pd.read_csv('./datasets/kcat/train/Kcatkm_train.csv')[0:100]
    df_val = pd.read_csv('./datasets/kcat/train/Kcatkm_val.csv')[0:10]
    df_test = pd.read_csv('./datasets/kcat/train/Kcatkm_test.csv')[0:10]

    # 计算权重
    labels = np.log10(df_data["Y"].values)
    if cfg.SOLVER.loss_balance == 'CBW':
        weights = Smooth_Label_CBW(labels)
    elif cfg.SOLVER.loss_balance == 'CSW':
        weights = Smooth_Label_CSW(labels)
    elif cfg.SOLVER.loss_balance == 'DMW':
        weights = Smooth_Label_DMW(labels)
    elif cfg.SOLVER.loss_balance == 'LDS':
        weights = Smooth_Label_LDS(labels)
    else:
        weights = np.ones(len(labels))

    weights_df = pd.DataFrame(weights, columns=['w'])
    df_data = pd.concat([df_data, weights_df], axis=1)

    # 生成数据格式
    train_dataset = DTIDataset(df_train.index.values, df_train,cfg)
    val_dataset = DTIDataset(df_val.index.values, df_val,cfg)
    test_dataset = DTIDataset(df_test.index.values, df_test,cfg)
    if cfg.protein_encode.input =='integer':
        graph_collate_func = integer_graph_collate_func
    else:
        graph_collate_func = prottrans_graph_collate_func

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}


    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    ### LR_decay
    if cfg.SOLVER.scheduler=='StepLR':
        step_size = cfg.SOLVER.StepLR_step_size
        gamma = cfg.SOLVER.StepLR_gamma
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    elif cfg.SOLVER.scheduler=='CosineAnnealingLR':
        T_max = cfg.SOLVER.CosineAnnealingLR_T_max
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
    else:
        scheduler = None


    torch.backends.cudnn.benchmark = True


    trainer = Trainer(model, opt, scheduler, device, training_generator, val_generator, test_generator,
                      discriminator=None,
                      experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
