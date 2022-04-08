import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
# import numpy as np
# import os
# import torch.nn as nn
# from utils.torch_util import EarlyStopping
# from utils.general_util import loss_plot
# #import mlflow

# class Train:
#     def __init__(self, param, model_param):

#         out_dir = param['out_dir']
#         batch_size = param['batch_size']
#         earlystop = param['earlystop']
#         EPOCH = param['EPOCH']
#         device = param['device']
#         criterion = nn.MSELoss(reduction="mean")

#         framework = model_param['model']
#         optimizer = model_param['optimizer']
#         scheduler = model_param['scheduler']
#         swa_start = model_param['swa_start']
#         swa_model = model_param['swa_model']
#         swa_sch = model_param['swa_scheduler']

#         #neuType = model_param['neuType']
#         #beta1 = model_param['beta1']
#         #beta2 = model_param['beta2']
#         #beta3 = model_param['beta3']

#     def run(self, loader):
        
#         best_model = os.path.join(f"{out_dir}/checkpoints/best_model.pth")
#         torch.save(framework.state_dict(), best_model)

#         plot_fig = os.path.join(f"{out_dir}/visualize/latest.png")
#         # early stopping patience; how long to wait after last time validation loss improved.
#         early_stopping = EarlyStopping(
#             patience=earlystop, verbose=True, path=best_model
#         )
#         avg_train_losses = []
#         avg_valid_losses = []
#         val_corr = []

#         for epoch in range(EPOCH):
            
#             avg_train_loss = train(loader['train'], epoch)
#             avg_valid_loss, corr = validate(loader['valid'])

#             avg_train_losses.append(avg_train_loss)
#             avg_valid_losses.append(avg_valid_loss)
#             val_corr.append(corr)

#             if epoch > swa_start:
#                 swa_model.update_parameters(framework)
#                 swa_sch.step()
#             else:
#                 scheduler.step()

#             early_stopping(np.median(avg_valid_losses), framework)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#         loss_plot(
#             train_loss=avg_train_losses, valid_loss=avg_valid_losses, file=plot_fig
#         )
#         # Update bn statistics for the swa_model at the end
#         torch.optim.swa_utils.update_bn(loader['train'], swa_model) #error
        
#         torch.save(swa_model.state_dict(), os.path.join(f"{out_dir}/checkpoints/swa_best_model.pth"))
#         torch.save(framework.state_dict(), best_model)
#         #framework.load_state_dict(torch.load(best_model))
        
#         #return early_stopping.val_loss_min
#         return val_corr.pop()

def train(loader, params, cfg, epoch):

    eval = {"predicted_value": list(), "real_value": list()}
    params["model"].train()
    train_loss = 0.0

    for batch_idx, data in enumerate(loader):

        # Limiting training data for faster epochs.
        if batch_idx * cfg["batch_size"] >= 30:
            break

        X, Y = data["X"].to(cfg["device"]), data["Y"].to(cfg["device"])
        params["optimizer"].zero_grad()
        pred = params["model"](X)
        loss = F.smooth_l1_loss(pred, Y)
        
        loss.backward()
        params["optimizer"].step()

        eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
        eval["real_value"] += Y.cpu().detach().numpy().tolist()
        train_loss += loss.item()
        idx = batch_idx + 1

        if batch_idx % 50 == 0:
            #mlflow.log_text(f"Training step : Epoch : [{epoch}/{EPOCH}], [{batch_idx}/{batch_size}, Loss_reg : {loss}")
            print(f"Training step : Epoch : [{epoch}/{cfg['EPOCH']}], [{batch_idx}/{cfg['batch_size']}, Loss_reg : {loss}")

    corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
    print(f"Training Spearman correlation = {corrs}")
    avg_train_loss = train_loss / idx

    return avg_train_loss

def validate(loader, params, cfg):

    eval = {"predicted_value": list(), "real_value": list()}
    params["model"].eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            
            X, Y = data['X'].to(cfg["device"]), data['Y'].to(cfg["device"])
            pred = params["model"](X)
            loss = F.smooth_l1_loss(pred, Y)

            eval["predicted_value"] += pred.cpu().detach().numpy().tolist()
            eval["real_value"] += Y.cpu().detach().numpy().tolist()
            val_loss += loss.item()
            idx = batch_idx + 1

    corrs = spearmanr(eval["real_value"], eval["predicted_value"])[0]
    print(f"Validation Spearman correlation = {corrs}")
    avg_val_loss = val_loss / idx

    return avg_val_loss, corrs


def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
