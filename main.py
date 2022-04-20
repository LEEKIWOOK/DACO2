import os
import sys
import yaml
import warnings
import argparse

import mlflow
import optuna
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from sklearn.model_selection import KFold

from model.unimodal.gRNA.embd_conv import EMBD_MK_CNN

from data.data_load import data_read, data_loader
from training.train import train, validate, reset_weights
from utils.general_util import make_mlflow_callback, loss_plot
from utils.adamw import AdamW
from utils.torch_util import EarlyStopping

warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, args):
        config_file = "./config.yaml"
        with open(config_file) as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)
        data_cfg = config["DATA"]
        model_cfg = config["MODEL"]

        data_list = ["cas9_kim", "cas9_wang", "cas9_xiang", "cas12a_kim"]
        idx = 0 if int(args.target) == 1 else 1 if int(args.target) >= 2 and int(args.target) <= 4 else 2 if int(args.target) >= 5 and int(args.target) <= 7 else 3 if int(args.target) == 8 else -1
        if idx < 0:
            print(f"target parameter : (1:Kim-Cas9, 2:Wang-wt, 3:Wang-HF1, 4:Wang-esp1, 5:Xiang-D2, 6:Xiang-D8, 7:Xiang-D10, 8:Kim-Cas12a)")
            sys.exit(-1)

        self.cv_results = {}
        self.cfg = {
            
            'kfold' : int(model_cfg["kfold"]),
            'target' : int(args.target),
            'seqidx' : idx,
            'seqpath' : f"%s/%s" % (data_cfg['in_dir'], data_cfg[data_list[idx]]),
            'earlystop' : int(model_cfg["earlystop"]),
            'batch_size' : int(model_cfg["batch"]),
            'EPOCH' : int(model_cfg["epoch"]),
            'seed' : int(model_cfg["seed"]),
            'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            
            'embd' : int(args.embd), #int(model_cfg["embd"]), #0: onehot / 1: embd_table / 2: word2vec
            'kmer' : int(args.kmer), #int(model_cfg["kmer"])
            'stride' : int(args.stride), #int(model_cfg["stride"])

            'dna2vec_path' : data_cfg['w2v_model'],
            'rna2_mod' : False if args.model.find('2') else True,
            'chro_mod' : False if args.model.find('3') else True,           
        }
        pam_len = 4 if self.cfg['target'] == 8 else 3
        
        if args.seqinfo == "spacer":
            self.cfg['seqlen'] = 20 + pam_len
        elif args.seqinfo == "full":
            if args.target == 8:
                self.cfg['seqlen'] = 34
            elif args.target >= 2 and args.target <= 4:
                self.cfg['seqlen'] = 23
            else:
                self.cfg['seqlen'] = 30
        else:
            seqinfo = list(args.seqinfo)
            if seqinfo[0] == 'd' and int(seqinfo[1]) >= 1 and int(seqinfo[1]) <= 5:
                self.cfg['seqlen'] = 20 - int(seqinfo[1]) + pam_len #
            if seqinfo[0] == 'i' and int(seqinfo[1]) >= 1 and int(seqinfo[1]) <= 3:
                self.cfg['seqlen'] = 20 + (int(seqinfo[1]) * 2) + pam_len

        self.cfg['outdir'] = f"../output/embd/embd{self.cfg['embd']}_k{self.cfg['kmer']}_s{self.cfg['stride']}_seqlen{self.cfg['seqlen']}_data{self.cfg['target']}" if args.embd == 2 else f"../output/embd/embd{self.cfg['embd']}_seqlen{self.cfg['seqlen']}_data{self.cfg['target']}"
        os.makedirs(f"{self.cfg['outdir']}/visualize", exist_ok=True)
        os.makedirs(f"{self.cfg['outdir']}/checkpoints", exist_ok=True)
        torch.manual_seed(self.cfg['seed'])

    def define_hyperparam(self):
    #def define_hyperparam(self, trial):
        print("define_hyperparam.")

        params = {
            #"criterion": nn.MSELoss(reduction = "mean"),
            #"criterion": F.smooth_l1_loss(),
            #"dropprob" : trial.suggest_float("dropprob", 3e-1, 4e-1),
            "dropprob" : 3e-1,
            #"lr": trial.suggest_float("lr", 1e-2, 3e-2, log=True),
            "lr" : 1e-2,
            #"weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
            "weight_decay": 3e-5,
            #"swa_lr": trial.suggest_float("swa_lr", 5e-3, 1e-2, log=True),
            "swa_lr": 5e-3,
            "swa_start": 5,
            #"swa_start": trial.suggest_categorical("swa_start", [5, 25]),
        }
        params["model"] = EMBD_MK_CNN(params["dropprob"], self.cfg).to(self.cfg['device'])
        params["model"].apply(reset_weights)

        params['optimizer'] = AdamW(params["model"].parameters(), lr = params["lr"], weight_decay=params["weight_decay"])
        params['swa_model'] = optim.swa_utils.AveragedModel(params["model"])
        params['scheduler'] = optim.lr_scheduler.CosineAnnealingLR(params["optimizer"], T_max = self.cfg['EPOCH'])
        params["swa_scheduler"] = optim.swa_utils.SWALR(optimizer=params["optimizer"], swa_lr = params["swa_lr"])
        
        return params
    
    def iteration(self, params, train_loader, valid_loader): #Fold

        best_model = os.path.join(f"{self.cfg['outdir']}/checkpoints/best_model.pth")
        #torch.save(params["model"].state_dict(), best_model)
        plot_fig = os.path.join(f"{self.cfg['outdir']}/visualize/latest.png")
        tloss_list, vloss_list, corr_list = list(), list(), list()

        early_stopping = EarlyStopping(patience=self.cfg["earlystop"], verbose=True, path = best_model)

        for epoch in range(0, self.cfg["EPOCH"]):
            
            print(f"Run : epoch {epoch+1} / {self.cfg['EPOCH']}...")
            tloss = train(train_loader, params, self.cfg, epoch) #params, cfg parameters
            vloss, corr = validate(valid_loader, params, self.cfg) #params, cfg parameters

            tloss_list.append(tloss)
            vloss_list.append(vloss)
            corr_list.append(corr)

            if epoch > params["swa_start"]:
                params["swa_model"].update_parameters(params["model"])
                params["swa_scheduler"].step()
            else:
                params["scheduler"].step()
            
            early_stopping(np.median(vloss_list), params["model"])
            if early_stopping.early_stop:
                print("Early stopping")
                break

        loss_plot(train_loss = tloss_list, valid_loss = vloss_list, file=plot_fig)
        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(train_loader, params["swa_model"]) #error
        
        torch.save(params["swa_model"].state_dict(), os.path.join(f"{self.cfg['outdir']}/checkpoints/swa_best_model.pth"))
        torch.save(params["model"].state_dict(), best_model)
        #self.framework.load_state_dict(torch.load(best_model))
        
        return corr_list.pop() #last correlation

    def objective_cv(self):
    #def objective_cv(self, trial):
        
        data = data_read(self.cfg)
        fold = KFold(n_splits = self.cfg["kfold"], shuffle=True)
        params = self.define_hyperparam() #(trial)

        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data)))):
            print("----------------------------------------------------------------------")
            print(f"FOLD : {fold_idx}")
            train_loader, valid_loader = data_loader(self.cfg, data, train_idx, valid_idx)
            
            with mlflow.start_run():
                corr = self.iteration(params, train_loader, valid_loader) #model 
                #mlflow.log_params(trial.params)
                self.cv_results[fold_idx] = corr

    def print_cv_results(self):

        print("\n++++++++++++++++++++++++++++++++++\n")
        print(f"K-FOLD CROSS VALIDATION RESULTS FOR {self.cfg['kfold']} FOLDS")
        print("\n++++++++++++++++++++++++++++++++++\n")
        sum = 0.0
        for key, value in self.cv_results.items():
            print(f'Fold {key}: {value}')
            sum += value
        print(f'Average: {sum/len(self.cv_results.items())}')
        print("\n++++++++++++++++++++++++++++++++++\n")

    def hyperparameter_tunning(self, study):

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("\n++++++++++++++++++++++++++++++++++\n")

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("\n++++++++++++++++++++++++++++++++++\n")

        print("Best trial:")
        trial = study.best_trial
        print("  Trial number: ", trial.number)
        print("  Sp corr. (trial value): ", trial.value)
        print("\n++++++++++++++++++++++++++++++++++\n")
        
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print("\n++++++++++++++++++++++++++++++++++\n")
        df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
        df.to_csv(f'{self.out_dir}/optuna_result.csv', sep='\t', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, help="1:Kim-Cas9, 2:Wang-wt, 3:Wang-HF1, 4:Wang-esp1, 5:Xiang-D2, 6:Xiang-D8, 7:Xiang-D10, 8:Kim-Cas12a", required=True)
    parser.add_argument("--model", type=str, help="gRNA module, add RNAss module : 1, Cromatin information : 2, .more than one input is possible (e.g. 1,2)", default='0')
    parser.add_argument("--embd", type=int, help="0:onehot encoding, 1:embd table, 2:word2vec")
    parser.add_argument("--seqinfo", type=str, default='spacer', help="spacer=20+pam, full=34, d[1-5], i[1-3]")
    parser.add_argument("--kmer", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    runner = Runner(args)
    #-> validate spearman correlation -> maximization
    #study = optuna.create_study(study_name="DACO - gRNA module : parameter optimization", direction="maximize",  pruner=optuna.pruners.HyperbandPruner(max_resource="auto")) #
    #mlflow_cb = make_mlflow_callback(runner.cfg['outdir'])
    
    #study.optimize(runner.objective_cv, n_trials = 1, timeout = 1200, callbacks=[mlflow_cb]) #hyperparameter tunning
    runner.objective_cv()
    #runner.hyperparameter_tunning(study)
    runner.print_cv_results()