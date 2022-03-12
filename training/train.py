import os
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_util import EarlyStopping, ForeverDataIterator
from utils.general_util import loss_plot

class Train:
    def __init__(self, param):

        self.earlystop = param['earlystop']
        self.EPOCH = param['EPOCH']
        self.device = param['device']
        self.criterion = nn.MSELoss(reduction="mean")

    def run(self, model, loader, logger):

        framework = model['model']
        optimizer = model['optimizer']
        scheduler = model['scheduler']
        swa_start = model['swa_start']
        swa_model = model['swa_model']
        swa_sch = model['swa_scheduler']

        best_model = os.path.join(logger.get_checkpoint_path("latest"))
        torch.save(framework.state_dict(), os.path.join(best_model + "_net.pth"))
        plot_fig = logger.get_image_path("latest.png")
        # early stopping patience; how long to wait after last time validation loss improved.
        early_stopping = EarlyStopping(
            patience=self.earlystop, verbose=True, path=best_model
        )
        avg_train_losses = []
        avg_valid_losses = []

        for epoch in range(self.EPOCH):
            
            avg_train_loss = self.train(framework, optimizer, loader['train'], epoch)
            avg_valid_loss = self.validate(framework, loader['valid'], epoch)

            avg_train_losses.append(avg_train_loss)
            avg_valid_losses.append(avg_valid_loss)

            if epoch > swa_start:
                swa_model.update_parameters(framework)
                swa_sch.step()
            else:
                scheduler.step()

            early_stopping(np.median(avg_valid_losses), framework)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        loss_plot(
            train_loss=avg_train_losses, valid_loss=avg_valid_losses, file=plot_fig
        )
        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(loader['train'], swa_model)
        # Use swa_model to make predictions on test data 
        #preds = swa_model(test_input)
        # load the last checkpoint with the best model
        framework.load_state_dict(torch.load(best_model + "_net.pth"))

        return best_model, early_stopping.val_loss_min

    def train(self, framework, optimizer, loader, epoch):

        eval = {"predicted_value": list(), "real_value": list()}
        framework.train()
        train_loss = 0.0
        #len_iter = len(iter)

        for batch_idx, data in enumerate(loader):

            dnaseq, rnass, rnamat, target = data['window'].to(self.device), data['rnass'].to(self.device), data['rnamat'].to(self.device), data['efficiency'].to(self.device)
            optimizer.zero_grad()
            pred = framework(dnaseq, rnass, rnamat)
            loss = self.criterion(pred, target)
            loss.backward()
            optimizer.step()

            eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
            eval["real_value"] += target.cpu().detach().numpy().tolist()
            train_loss += loss.item()
            idx = batch_idx + 1

            # if batch_idx % 10 == 0:
            #     batch_size = len(data)
            #     print(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{set_size} "
            #         f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}")
            #     #print(f"Training step : Epoch : [{epoch}/{self.EPOCH}] [{i}/{len(self.train_target_iter)}], Loss_reg : {loss}")

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Training Spearman correlation = {corrs}")
        avg_train_loss = train_loss / idx

        return avg_train_loss

    def validate(self, framework, loader, epoch):
        
        eval = {"predicted_value": list(), "real_value": list()}
        framework.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                dnaseq, rnass, rnamat, target = data['window'].to(self.device), data['rnass'].to(self.device), data['rnamat'].to(self.device), data['efficiency'].to(self.device)
                pred = framework(dnaseq, rnass, rnamat)
                loss = self.criterion(pred, target)

                eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
                eval["real_value"] += target.cpu().detach().numpy().tolist()
                val_loss += loss.item()
                idx = batch_idx + 1

        corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
        print(f"Validation Spearman correlation = {corrs}")
        avg_val_loss = val_loss / idx
        
        return avg_val_loss

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
