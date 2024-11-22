comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DrugBAN
from time import time
from utils import set_seed, mkdir, prottrans_graph_collate_func
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,pairwise_distances,r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="KcatBAN for kcat prediction")
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
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    output_dir = cfg.RESULT.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataFolder = f'./datasets/{args.data}'
    # dataFolder = os.path.join(dataFolder, str(args.split))

    # 读取数据
    data_path = os.path.join(dataFolder,'Indepent_data_clean_806.csv')
    df_data = pd.read_csv(data_path)
    df_data["Y"] = df_data["Y"].astype(float)
    labels = np.log10(df_data["Y"].values)
    weights = np.ones(len(labels))
    weights_df = pd.DataFrame(weights, columns=['w'])
    # df_data = pd.concat([df_data, weights_df], axis=1)
    df = pd.concat([df_data, weights_df], axis=1)
    # 随机化数据集
    # df = df_data.sample(frac=1, random_state=100)
    # 计算切分的索引
    total_rows = len(df)
    train_end_index = int(total_rows * 0.8)
    valid_end_index = int(total_rows * 0.9)
    #df_test = df.iloc[valid_end_index:]
    df_test = df
    #df_test.to_csv(os.path.join(dataFolder, 'independent_test_lamba0.2_seed10022.csv'))
    df_test = df_test.reset_index(drop=True)

    # df_test = pd.read_csv(r"F:\SecondStudy\DATA\Independent\trCsMSX.csv")

    test_dataset = DTIDataset(df_test.index.values, df_test, cfg)
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': False, 'collate_fn': prottrans_graph_collate_func}
    test_generator = DataLoader(test_dataset, **params)
    model = DrugBAN(**cfg).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load("./result/loss_lamba0.2_seed100/best_model_epoch_80.pth"))
    #model.load_state_dict(torch.load("./result/Ablation_study/revised_seed100_bestTest_FFF/best_model_epoch_90.pth"))

    # 将模型设置为评估模式
    model.eval()
    # opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    with torch.no_grad():
        test_loss = 0
        y_label, y_pred,att_list = [], [], []
        data_loader = test_generator
        num_batches = len(data_loader)
        with torch.no_grad():
            preds = []
            for i, (v_d, v_p, labels, weights) in enumerate(data_loader):
                v_d, v_p, labels, weights = v_d.to(device), v_p.to(device), \
                                            labels.float().to(device), weights.to(device)
                targets = labels.unsqueeze(1)
                v_d, v_p, score,att = model(v_d, v_p, mode = 'eval')

                att = att.data.cpu().numpy()
                att = att.tolist()
                att_list.extend(att)
                batch_preds = score.data.cpu().numpy()
                # Collect vectors
                batch_preds = batch_preds.tolist()
                preds.extend(batch_preds)
                means = score[:, [j for j in range(len(score[0])) if j % 4 == 0]]
                lambdas = score[:, [j for j in range(len(score[0])) if j % 4 == 1]]
                alphas = score[:, [j for j in range(len(score[0])) if j % 4 == 2]]
                betas = score[:, [j for j in range(len(score[0])) if j % 4 == 3]]
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + means.to("cpu").tolist()
        test_loss = test_loss / num_batches

        p = []
        y = []
        c = []
        var = []
        data_uncertainty_list = []
        model_uncertainty_list = []
        total_uncertainty_list = []
        drug_att_list = []
        protein_att_list = []
        for i in range(len(preds)):
            means = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 0])
            lambdas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 1])
            alphas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 2])
            betas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 3])
            # means, lambdas, alphas, betas = np.split(np.array(preds[i]), 4)

            inverse_evidence = 1. / ((alphas - 1) * lambdas)
            data_uncertainty = betas / (alphas - 1)
            model_uncertainty = betas / (alphas - 1) / lambdas
            total_uncertainty = data_uncertainty + model_uncertainty
            p.append(means)
            y.append(np.array([y_label[i]]))

            protein_att = np.mean(att_list[i],axis=1)
            protein_att = np.mean(protein_att,axis=0)
            drug_att = np.mean(att_list[i],axis=2)
            drug_att = np.mean(drug_att,axis=0)

            # NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
            # confidence. we can use this or the Var[X] defined by NIG.
            c.append(inverse_evidence)
            var.append(betas * inverse_evidence)
            data_uncertainty_list.append(data_uncertainty)
            model_uncertainty_list.append(model_uncertainty)
            total_uncertainty_list.append(total_uncertainty)
            drug_att_list.append(drug_att)
            protein_att_list.append(protein_att)
        mse = mean_squared_error(y_label, y_pred)
        mae = mean_absolute_error(y_label, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr = np.corrcoef(y_label, np.squeeze(y_pred))[0, 1]
        #pearson_corr = 1 - pairwise_distances([y_label, np.squeeze(y_pred)], metric='correlation')[0, 1]
        r2 = r2_score(y_label, y_pred)
        evidential_result = np.hstack(
            (y, p, c, var, data_uncertainty_list, model_uncertainty_list, total_uncertainty_list))

        print('test ' + " mse " + str(mse) + " mae " + str(mae) + " rmse " + str(rmse)
              + " pearson_corr " + str(pearson_corr) + " r2 " + str(r2))
        column_name = ['y', 'p', 'c', 'var', 'data_uncertainty', 'model_uncertainty', 'total_uncertainty']
        evidential_df = pd.DataFrame(evidential_result, columns=column_name)
        evidential_df.to_csv(os.path.join(output_dir, "independent_test_result.csv"))
        protein_att_list = np.array(protein_att_list)
        drug_att_list = np.array(drug_att_list)
        np.save('protein_att',protein_att_list)
        np.save('drug_att',drug_att_list)

        print()
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

        return test_loss, mse, mae, rmse, pearson_corr, r2, evidential_result


if __name__ == '__main__':
    s = time()
    test_loss, mse, mae, rmse, pearson_corr, r2, evidential_result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

