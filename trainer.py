import torch
import torch.nn as nn
import copy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,pairwise_distances,r2_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from tqdm import tqdm
from utils import evidential_loss, EvidentialLossSumOfSquares,EvidentialLossNLL


class Trainer(object):
    def __init__(self, model, optim, scheduler, device, train_dataloader, val_dataloader, test_dataloader,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.eviloss_lamba = config["SOLVER"]["loss_lamba"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        # self.loss_function = evidential_loss()
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]

        self.loss_function = config["SOLVER"]['loss_function']
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_mse = 10000

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_mse_epoch = [], []
        self.test_metrics = {}
        self.evidential_result = []
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "val_loss","mse","mae","rmse","pearson_corr","r2"]
        test_metric_header = ["# Best Epoch", "val_loss","mse","mae","rmse","pearson_corr","r2"]
        train_metric_header = ["# Epoch", "Train_loss","mse","mae","rmse","pearson_corr","r2"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss,mse,mae,rmse,pearson_corr,r2 = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + \
                  list(map(float2str, [train_loss,mse,mae,rmse,pearson_corr,r2]))
            if self.scheduler != None:
                self.scheduler.step()
                print(f"Learning Rate: {self.scheduler.get_last_lr()}")
            if self.experiment:
                self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            val_loss,mse,mae,rmse,pearson_corr,r2 = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch mse", mse, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch mae", mae, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch rmse", rmse, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch pearson_corr", pearson_corr, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch r2", r2, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + \
                      list(map(float2str, [val_loss,mse,mae,rmse,pearson_corr,r2]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_mse_epoch.append(mse)
            if mse <= self.best_mse:
                self.best_model = copy.deepcopy(self.model)
                self.best_mse = mse
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss),
                  " mse "+ str(mse) + " mae " + str(mae)+" rmse " + str(rmse)
                  +" pearson_corr " + str(pearson_corr)+" r2 " + str(r2))

        # 真正的test
        test_loss,mse,mae,rmse,pearson_corr,r2,evidential_result = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + \
                   list(map(float2str, [val_loss,mse,mae,rmse,pearson_corr,r2]))
        self.evidential_result = evidential_result
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss),
              " mse "+ str(mse) + " mae " + str(mae)+" rmse " + str(rmse)
                  +" pearson_corr " + str(pearson_corr)+" r2 " + str(r2))
        self.test_metrics["mse"] = mse
        self.test_metrics["mae"] = mae
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["rmse"] = rmse
        self.test_metrics["pearson_corr"] = pearson_corr
        self.test_metrics["r2"] = r2
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_mse", self.best_mse)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_mse", self.test_metrics["mse"])
            self.experiment.log_metric("test_mae", self.test_metrics["mae"])
            self.experiment.log_metric("test_rmse", self.test_metrics["rmse"])
            self.experiment.log_metric("test_pearson_corr", self.test_metrics["pearson_corr"])
            self.experiment.log_metric("test_r2", self.test_metrics["r2"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())
        column_name = ['y', 'p', 'c', 'var','data_uncertainty','model_uncertainty','total_uncertainty']
        evidential_df = pd.DataFrame(self.evidential_result,columns=column_name)
        evidential_df.to_csv(os.path.join(self.output_dir, "test_evidential.csv"))

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        preds = []
        loss_epoch = 0
        y_label, y_pred = [], []
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels,weights) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels,weights = v_d.to(self.device), v_p.to(self.device), \
                                       labels.float().to(self.device),weights.to(self.device)
            targets = labels.unsqueeze(1)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            means = score[:, [j for j in range(len(score[0])) if j % 4 == 0]]
            lambdas = score[:, [j for j in range(len(score[0])) if j % 4 == 1]]
            alphas = score[:, [j for j in range(len(score[0])) if j % 4 == 2]]
            betas = score[:, [j for j in range(len(score[0])) if j % 4 == 3]]
            if self.loss_function == 'ev_v1':
                loss = evidential_loss(means, lambdas, alphas, betas, targets,lamba=self.eviloss_lamba)
            elif self.loss_function=='ev_v2':
                loss_function = EvidentialLossNLL()
                loss = loss_function(means, lambdas, alphas, betas, targets, self.eviloss_lamba,1e-4)
            loss = torch.mul(weights.unsqueeze(1),loss)
            batch_preds = score.data.cpu().numpy()
            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

            # if self.n_class == 1:
            #     n, loss = binary_cross_entropy(score, labels)
            # else:
            #     n, loss = cross_entropy_logits(score, labels)
            loss = torch.sum(loss)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            y_label = y_label + labels.to("cpu").tolist()
            y_pred = y_pred + means.to("cpu").tolist()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        p = []
        c = []
        var = []
        data_uncertainty_list = []
        model_uncertainty_list = []
        total_uncertainty_list = []
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

            # NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
            # confidence. we can use this or the Var[X] defined by NIG.
            c.append(inverse_evidence)
            data_uncertainty_list.append(data_uncertainty)
            model_uncertainty_list.append(model_uncertainty)
            total_uncertainty_list.append(total_uncertainty)
        mse = mean_squared_error(y_label, y_pred)
        mae = mean_absolute_error(y_label, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr = 1 - pairwise_distances([y_label, np.squeeze(y_pred)], metric='correlation')[0, 1]
        r2 = r2_score(y_label, y_pred)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch,mse,mae,rmse,pearson_corr,r2


    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            preds=[]
            for i, (v_d, v_p, labels,weights) in enumerate(data_loader):
                v_d, v_p, labels,weights = v_d.to(self.device), v_p.to(self.device), \
                                           labels.float().to(self.device), weights.to(self.device)
                targets = labels.unsqueeze(1)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)

                batch_preds = score.data.cpu().numpy()
                # Collect vectors
                batch_preds = batch_preds.tolist()
                preds.extend(batch_preds)
                means = score[:, [j for j in range(len(score[0])) if j % 4 == 0]]
                lambdas = score[:, [j for j in range(len(score[0])) if j % 4 == 1]]
                alphas = score[:, [j for j in range(len(score[0])) if j % 4 == 2]]
                betas = score[:, [j for j in range(len(score[0])) if j % 4 == 3]]
                # loss = evidential_loss(means, lambdas, alphas, betas, targets, lamba=self.eviloss_lamba)
                if self.loss_function == 'ev_v1':
                    loss = evidential_loss(means, lambdas, alphas, betas, targets, lamba=self.eviloss_lamba)
                elif self.loss_function == 'ev_v2':
                    loss_function = EvidentialLossNLL()
                    loss = loss_function(means, lambdas, alphas, betas, targets, self.eviloss_lamba, 1e-4)
                    # loss = EvidentialLossNLL(means, lambdas, alphas, betas, targets, lam=self.eviloss_lamba)
                loss = torch.mul(weights.unsqueeze(1), loss)
                # if self.n_class == 1:
                #     n, loss = binary_cross_entropy(score, labels)
                # else:
                #     n, loss = cross_entropy_logits(score, labels)
                loss = torch.sum(loss)
                test_loss += loss.item()
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

            # NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
            # confidence. we can use this or the Var[X] defined by NIG.
            c.append(inverse_evidence)
            var.append(betas * inverse_evidence)
            data_uncertainty_list.append(data_uncertainty)
            model_uncertainty_list.append(model_uncertainty)
            total_uncertainty_list.append(total_uncertainty)
        mse = mean_squared_error(y_label, y_pred)
        mae = mean_absolute_error(y_label, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr = 1 - pairwise_distances([y_label, np.squeeze(y_pred)], metric='correlation')[0, 1]
        r2 = r2_score(y_label, y_pred)
        evidential_result = np.hstack((y,p,c,var,data_uncertainty_list,model_uncertainty_list,total_uncertainty_list))

        if dataloader=='val':
            return test_loss,mse,mae,rmse,pearson_corr,r2
        if dataloader == 'test':
            return test_loss, mse, mae, rmse, pearson_corr, r2, evidential_result

