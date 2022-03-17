import os
import yaml
import warnings
import argparse

import mlflow
import optuna
import torch
import torch.optim as optim
#from model.unimodal.gRNA.embd_conv import EMBD_MK_CNN
from model.unimodal.gRNA.embd_mhsa import EMBD_MHSA
#from model.unimodal.gRNA.word2vec_conv import W2V_MK_CNN

from data.data_manager import DataManager
from training.train import Train
from utils.general_util import make_mlflow_callback
from utils.adamw import AdamW
#from utils.cyclic_schedulers import CyclicLRWithRestarts

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
            "dropprob" : trial.suggest_float("dropprob", 3e-1, 4e-1),
            "lr": trial.suggest_float("lr", 1e-2, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
            "swa_lr": trial.suggest_float("swa_lr", 5e-3, 1e-2, log=True),
            "multi_head_num": trial.suggest_categorical("multi_head_num", [8, 16, 32])
            #"swa_start": trial.suggest_categorical("swa_start", [5, 25])
        }
        
        return params
    
    def define_model(self, param):
        
        model_dict = dict()
        #model_dict["model"] = EMBD_MK_CNN(param['dropprob'], len = 33).to(self.device)
        #model_dict["model"] = W2V_MK_CNN(param['dropprob'], len = 33).to(self.device)
        model_dict["model"] = EMBD_MHSA(param['dropprob'], len = 33).to(self.device)
        model_dict["optimizer"] = AdamW(model_dict["model"].parameters(), lr = param["lr"], weight_decay=param["weight_decay"])
        model_dict["swa_model"] = optim.swa_utils.AveragedModel(model_dict["model"])
        model_dict["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(model_dict["optimizer"], T_max = self.EPOCH)
        model_dict["swa_scheduler"] = optim.swa_utils.SWALR(optimizer=model_dict["optimizer"], swa_lr = param["swa_lr"])
        #model_dict["swa_start"] = param["swa_start"]
        model_dict["swa_start"] = 5
      
        return model_dict

    def train_model(self, model):

        ML = Train(self.static_param, model) 
        corr = ML.run(loader)

        return corr
    
    def objective(self, trial):
        
        best_val_corr = float('Inf')
        with mlflow.start_run():
            
            param = self.define_hyperparam(trial)
            mlflow.log_params(trial.params)

            #Initialize network
            model = self.define_model(param)

            #Train network
            best_val_corr = self.train_model(model)

        return best_val_corr

    def print_results(self, study):

        print("\n++++++++++++++++++++++++++++++++++\n")
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial #

        print("  Trial number: ", trial.number)
        print("  Sp corr. (trial value): ", trial.value)

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
    
    #-> validate spearman correlation -> maximization
    study = optuna.create_study(study_name="DACO - gRNA module : parameter optimization", direction="maximize",  pruner=optuna.pruners.HyperbandPruner(max_resource="auto"))
    mlflow_cb = make_mlflow_callback(runner.out_dir)
    
    study.optimize(runner.objective, n_trials=10, callbacks=[mlflow_cb])
    print("Number of finished trials: {}".format(len(study.trials)))

    runner.print_results(study)