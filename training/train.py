import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

def train(loader, params, cfg, epoch):

    eval = {"predicted_value": list(), "real_value": list()}
    params["model"].train()
    train_loss = 0.0

    for batch_idx, data in enumerate(loader):

        # if batch_idx * cfg["batch_size"] >= 30:
        #     print("Limiting training data for faster epochs.")
        #     break

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
