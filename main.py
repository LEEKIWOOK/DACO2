import os
import yaml
import warnings
import argparse

import optuna
import mlflow
import torch
import torch.optim as optim
from pprint import pformat

from data.data_manager import DataManager
from model.unimodal.gRNA.deepbind import ConvNet
from training.train import Train
from utils.general_util import CompleteLogger
from utils.adamw import AdamW
from utils.cyclic_schedulers import CyclicLRWithRestarts

warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, args):
        config_file = "./config.yaml"
        with open(config_file) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)
        data_cfg = config["DATA"]
        model_cfg = config["MODEL"]

        domain_list = [
            "cas9_wt_kim",
            "cas9_wt_wang",
            "cas9_wt_xiang",
            "cas9_hf_wang",
            "cas9_esp_wang",
            "cas12a_wt_kim",
        ]

        self.out_dir = f"{data_cfg['out_dir']}/data_{args.target}/set{args.set}/"
        self.batch_size = int(model_cfg["batch"])
        self.EPOCH = int(model_cfg["epoch"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_run = args.model.split(',')

        #data_manager parameter
        self.dm_param = {
            
            'model_run': self.model_run,
            'dna2vec_path' : data_cfg['w2v_model'],
            'out_dir' : self.out_dir,
            'seq_prefix' : f"%s/%s" % (data_cfg['in_dir'], data_cfg[domain_list[args.target]]),
            'rna_prefix' : f"%s/%s" % (data_cfg['rna_dir'], data_cfg[domain_list[args.target]]),
            'chro_prefix' : f"%s/%s" % (data_cfg['chr_dir'], data_cfg[domain_list[args.target]]),
            'target_domain' : args.target,
            'kmer' : int(model_cfg["kmer"]),
            'batch_size' : self.batch_size,
            'seed' : int(model_cfg["seed"])
        }

        #training parameter
        self.train_param  = {
            'earlystop' : int(model_cfg["earlystop"]),
            'EPOCH' : self.EPOCH,
            'device' : self.device
        }

    def dataload(self):
            
        DM = DataManager(self.dm_param)
        os.makedirs(self.out_dir, exist_ok=True) #

        loaderset = DM.data_load() #DataLoader 구분
        return loaderset

    def define_hyperparam(self, trial):
        #print("define_hyperparam.")
        model_params = {
            "pool": trial.suggest_categorical("pool", ["max", "maxavg"]),
            "sigmaConv": trial.suggest_float("sigmaConv", 10**-7, 10**-3),
            "sigmaNeu": trial.suggest_float("sigmaNeu", 10**-5, 10**-2),
            "dropprob" : trial.suggest_float("dropprob", 0.0, 0.5, step=0.1),
            "hnode" : trial.suggest_categorical("hnode", [16, 32, 64]),
            "neuType": trial.suggest_categorical("neuType", ["hidden", "nohidden"]),
        }
        learning_params = {
            "lr": trial.suggest_float("lr", 0.0005,0.05, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
            "momentum": trial.suggest_float("momentum", 0.95,0.99),
            "scheduler": trial.suggest_categorical("scheduler", ["CyclicLR", "CosineLR"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3),
            "t_mult": trial.suggest_float("t_mult", 1.0, 3.0),

            "beta1": trial.suggest_float("beta1", 10**-15, 10**-3),
            "beta2": trial.suggest_float("beta2", 10**-15, 10**-3),
            "beta3": trial.suggest_float("beta3", 10**-15, 10**-3),
        }
        
        print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
        return model_params, learning_params
    
    def define_model(self, param_m, param_l):
        
        model_dict = dict()
        model_dict["model"] = ConvNet(param_m, glen = 33).to(self.device)

        if param_l["optimizer"] == "AdamW":
            model_dict["optimizer"] = AdamW([model_dict["model"].wConv,model_dict["model"].wRect,model_dict["model"].wNeu,model_dict["model"].wNeuBias,model_dict["model"].wHidden,model_dict["model"].wHiddenBias], lr = param_l["lr"], weight_decay=param_l["weight_decay"])
        elif param_l["optimizer"] == "SGD":
            model_dict["optimizer"] = optim.SGD([model_dict["model"].wConv,model_dict["model"].wRect,model_dict["model"].wNeu,model_dict["model"].wNeuBias,model_dict["model"].wHidden,model_dict["model"].wHiddenBias], lr = param_l["lr"], momentum=param_l["momentum"])

        if param_l["scheduler"] == "CyclicLR":
            model_dict["scheduler"] = CyclicLRWithRestarts(optimizer = model_dict["optimizer"], batch_size = self.batch_size, epoch_size=self.EPOCH, restart_period=3, policy="cosine", t_mult=param_l["t_mult"])
        elif param_l["scheduler"] == "CosineLR":
            model_dict["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(model_dict["optimizer"], T_max = self.EPOCH)

        model_dict["swa_start"] = int(self.EPOCH / 20)
        model_dict["swa_model"] = optim.swa_utils.AveragedModel(model_dict["model"])
        model_dict["swa_scheduler"] = optim.swa_utils.SWALR(optimizer=model_dict["optimizer"], swa_lr = param_l["lr"])

        model_dict["neuType"] = param_m["neuType"]
        model_dict["beta1"] = param_l["beta1"]
        model_dict["beta2"] = param_l["beta2"]
        model_dict["beta3"] = param_l["beta3"]
      
        return model_dict

    def train_model(self, model, loader, logger):

        ML = Train(self.train_param, model) 
        loss = ML.run(loader, logger)

        return loss
    
    def objective(self, trial):
        
        logger = CompleteLogger(self.out_dir)
        print("\n********************************\n")
        mlflow.set_experiment('task_220311')

        best_val_loss = float('Inf')

        with mlflow.start_run():
            param_m, param_l = self.define_hyperparam(trial)
            mlflow.log_params(trial.params)
            mlflow.log_param("device", self.device)

            #Get DataLoader
            loader = self.dataload()
        
            #Initialize network
            model = self.define_model(param_m, param_l)

            #Train network
            best_val_loss = self.train_model(model, loader, logger)
        
        # Return the best validation loss achieved by the network.
        # This is needed as Optuna needs to know how the suggested hyperparameters are influencing the network loss.
        logger.close()

        return best_val_loss

    def print_results(self, study):

        print("\n++++++++++++++++++++++++++++++++++\n")
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Trial number: ", trial.number)
        print("  Loss (trial value): ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ == "__main__":
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(study_name="XAI - Crispr", direction="minimize")

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, help="Training file (1 ~ 5)", required=True)
    parser.add_argument("--model", type=str, help="gRNA:1, RNAss:2, CA:3, .more than one input is possible (e.g. 1,2)", required=True)
    parser.add_argument("--set", type=int, help=">1", default=1)
    args = parser.parse_args()

    runner = Runner(args)
    study.optimize(runner.objective, n_trials=10)

    # Print optuna study statistics
    runner.print_results(study)

    