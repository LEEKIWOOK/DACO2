import os
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
from utils.torch_util import EarlyStopping
from utils.general_util import loss_plot
#import mlflow

class Train:
    def __init__(self, param, model_param):

        self.out_dir = param['out_dir']
        self.batch_size = param['batch_size']
        self.earlystop = param['earlystop']
        self.EPOCH = param['EPOCH']
        self.device = param['device']
        self.criterion = nn.MSELoss(reduction="mean")

        self.framework = model_param['model']
        self.optimizer = model_param['optimizer']
        self.scheduler = model_param['scheduler']
        self.swa_start = model_param['swa_start']
        self.swa_model = model_param['swa_model']
        self.swa_sch = model_param['swa_scheduler']

        #self.neuType = model_param['neuType']
        #self.beta1 = model_param['beta1']
        #self.beta2 = model_param['beta2']
        #self.beta3 = model_param['beta3']

    def run(self, loader):
        
        best_model = os.path.join(f"{self.out_dir}/checkpoints/best_model.pth")
        torch.save(self.framework.state_dict(), best_model)

        plot_fig = os.path.join(f"{self.out_dir}/visualize/latest.png")
        # early stopping patience; how long to wait after last time validation loss improved.
        early_stopping = EarlyStopping(
            patience=self.earlystop, verbose=True, path=best_model
        )
        avg_train_losses = []
        avg_valid_losses = []
        val_corr = []

        for epoch in range(self.EPOCH):
            
            avg_train_loss = self.train(loader['train'], epoch)
            avg_valid_loss, corr = self.validate(loader['valid'])

            avg_train_losses.append(avg_train_loss)
            avg_valid_losses.append(avg_valid_loss)
            val_corr.append(corr)

            if epoch > self.swa_start:
                self.swa_model.update_parameters(self.framework)
                self.swa_sch.step()
            else:
                self.scheduler.step()

            early_stopping(np.median(avg_valid_losses), self.framework)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        loss_plot(
            train_loss=avg_train_losses, valid_loss=avg_valid_losses, file=plot_fig
        )
        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(loader['train'], self.swa_model) #error
        
        torch.save(self.swa_model.state_dict(), os.path.join(f"{self.out_dir}/checkpoints/swa_best_model.pth"))
        torch.save(self.framework.state_dict(), best_model)
        #self.framework.load_state_dict(torch.load(best_model))
        
        #return early_stopping.val_loss_min
        return val_corr.pop()

    def train(self, loader, epoch):

        eval = {"predicted_value": list(), "real_value": list()}
        self.framework.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(loader):

            Xg, Yg = data['Xg'].to(self.device), data['Yg'].to(self.device)
            self.optimizer.zero_grad()
            pred = self.framework(Xg)
            loss = self.criterion(pred, Yg)
            
            loss.backward()
            self.optimizer.step()

            eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
            eval["real_value"] += Yg.cpu().detach().numpy().tolist()
            train_loss += loss.item()
            idx = batch_idx + 1

            if batch_idx % 50 == 0:
                #mlflow.log_text(f"Training step : Epoch : [{epoch}/{self.EPOCH}], [{batch_idx}/{self.batch_size}, Loss_reg : {loss}")
                print(f"Training step : Epoch : [{epoch}/{self.EPOCH}], [{batch_idx}/{self.batch_size}, Loss_reg : {loss}")

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Training Spearman correlation = {corrs}")
        avg_train_loss = train_loss / idx

        return avg_train_loss

    def validate(self, loader):
        
        eval = {"predicted_value": list(), "real_value": list()}
        self.framework.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                
                Xg, Yg = data['Xg'].to(self.device), data['Yg'].to(self.device)
                pred = self.framework(Xg)
                loss = self.criterion(pred, Yg)

                eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
                eval["real_value"] += Yg.cpu().detach().numpy().tolist()
                val_loss += loss.item()
                idx = batch_idx + 1

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Validation Spearman correlation = {corrs}")
        avg_val_loss = val_loss / idx
        
        return avg_val_loss, corrs

    # def test(self):

    #     eval = {"predicted_value": list(), "real_value": list()}
    #     self.framework.eval()
    #     with torch.no_grad():
    #         for i in range(len(self.test_target_iter)):
                
    #             GN, GE, CX, Y = next(self.test_target_iter)
    #             GN, GE, CX, Y = GN.to(self.device), GE.to(self.device), CX.to(self.device), Y.to(self.device)

    #             pred = self.framework(GN, GE, CX)

    #             eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
    #             eval["real_value"] += Y.cpu().detach().numpy().tolist()

    #     corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
    #     corrp = pearsonr(eval["real_value"], eval["predicted_value"])[0]

    #     print(f"Spearman Correlation.\t{corrs}")
    #     print(f"Pearson Correlation.\t{corrp}")
