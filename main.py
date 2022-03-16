import os
import yaml
import warnings
import argparse

import mlflow
import optuna
#from optuna.integration.mlflow import MLflowCallback
#from optuna.trial import TrialState

import torch
import torch.optim as optim
#import joblib as joblib
#from pprint import pformat

from data.data_manager import DataManager
from model.unimodal.gRNA.deepbind import ConvNet
from training.train import Train
from utils.general_util import make_mlflow_callback
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
        os.makedirs(f"{self.out_dir}/visualize", exist_ok=True)
        os.makedirs(f"{self.out_dir}/checkpoints", exist_ok=True)

        self.batch_size = int(model_cfg["batch"])
        self.EPOCH = int(model_cfg["epoch"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_run = args.model.split(',')

        #data_manager parameter
        self.static_param = {
            
            'model_run': self.model_run,
            'dna2vec_path' : data_cfg['w2v_model'],
            'out_dir' : self.out_dir,
            'seq_prefix' : f"%s/%s" % (data_cfg['in_dir'], data_cfg[domain_list[args.target]]),
            'rna_prefix' : f"%s/%s" % (data_cfg['rna_dir'], data_cfg[domain_list[args.target]]),
            'chro_prefix' : f"%s/%s" % (data_cfg['chr_dir'], data_cfg[domain_list[args.target]]),
            'target_domain' : args.target,
            'kmer' : int(model_cfg["kmer"]),
            'batch_size' : self.batch_size,
            'seed' : int(model_cfg["seed"]),
        
            'earlystop' : int(model_cfg["earlystop"]),
            'EPOCH' : self.EPOCH,
            'device' : self.device
        }

    def dataload(self):
            
        DM = DataManager(self.static_param)
        os.makedirs(self.out_dir, exist_ok=True) #

        loaderset = DM.data_load() #DataLoader 구분
        return loaderset

    def define_hyperparam(self, trial):
        #print("define_hyperparam.")
        params = {
            "pool": trial.suggest_categorical("pool", ["max", "maxavg"]),
            "sigmaConv": trial.suggest_float("sigmaConv", 10**-7, 10**-3),
            "sigmaNeu": trial.suggest_float("sigmaNeu", 10**-5, 10**-2),
            "dropprob" : trial.suggest_float("dropprob", 0.0, 0.5, step=0.1),
            "hnode" : trial.suggest_categorical("hnode", [16, 32, 64]),
            "neuType": trial.suggest_categorical("neuType", ["hidden", "nohidden"]),
        
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
        
        #print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
        return params
    
    def define_model(self, param):
        
        model_dict = dict()
        model_dict["model"] = ConvNet(param, glen = 33).to(self.device)

        if param["optimizer"] == "AdamW":
            model_dict["optimizer"] = AdamW([model_dict["model"].wConv,model_dict["model"].wRect,model_dict["model"].wNeu,model_dict["model"].wNeuBias,model_dict["model"].wHidden,model_dict["model"].wHiddenBias], lr = param["lr"], weight_decay=param["weight_decay"])
        elif param["optimizer"] == "SGD":
            model_dict["optimizer"] = optim.SGD([model_dict["model"].wConv,model_dict["model"].wRect,model_dict["model"].wNeu,model_dict["model"].wNeuBias,model_dict["model"].wHidden,model_dict["model"].wHiddenBias], lr = param["lr"], momentum=param["momentum"])

        if param["scheduler"] == "CyclicLR":
            model_dict["scheduler"] = CyclicLRWithRestarts(optimizer = model_dict["optimizer"], batch_size = self.batch_size, epoch_size=self.EPOCH, restart_period=3, policy="cosine", t_mult=param["t_mult"])
        elif param["scheduler"] == "CosineLR":
            model_dict["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(model_dict["optimizer"], T_max = self.EPOCH)

        model_dict["swa_start"] = int(self.EPOCH / 20)
        model_dict["swa_model"] = optim.swa_utils.AveragedModel(model_dict["model"])
        model_dict["swa_scheduler"] = optim.swa_utils.SWALR(optimizer=model_dict["optimizer"], swa_lr = param["lr"])

        model_dict["neuType"] = param["neuType"]
        model_dict["beta1"] = param["beta1"]
        model_dict["beta2"] = param["beta2"]
        model_dict["beta3"] = param["beta3"]
      
        return model_dict

    def train_model(self, model):

        ML = Train(self.static_param, model) 
        loss = ML.run(loader)

        return loss
    
    def objective(self, trial):
        
        best_val_loss = float('Inf')
        with mlflow.start_run():
            
            param = self.define_hyperparam(trial)
            mlflow.log_params(trial.params)

            #Initialize network
            model = self.define_model(param)

            #Train network
            best_val_loss = self.train_model(model)
            # Return the best validation loss achieved by the network.
            # This is needed as Optuna needs to know how the suggested hyperparameters are influencing the network loss.

        return best_val_loss

    def print_results(self, study):

        print("\n++++++++++++++++++++++++++++++++++\n")
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial #

        print("  Trial number: ", trial.number)
        print("  Loss (trial value): ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
        df.to_csv(f'{self.out_dir}/optuna_result.csv', sep='\t', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, help="Training file (1 ~ 5)", required=True)
    parser.add_argument("--model", type=str, help="gRNA:1, RNAss:2, CA:3, .more than one input is possible (e.g. 1,2)", required=True)
    parser.add_argument("--set", type=int, help=">1", default=1)
    args = parser.parse_args()

    runner = Runner(args)
    loader = runner.dataload()
    
    study = optuna.create_study(study_name="DACO - gRNA module : parameter optimization", direction="minimize",  pruner=optuna.pruners.HyperbandPruner(max_resource="auto"))
    mlflow_cb = make_mlflow_callback(runner.out_dir)
    
    study.optimize(runner.objective, n_trials=10, callbacks=[mlflow_cb])
    print("Number of finished trials: {}".format(len(study.trials)))

    runner.print_results(study)