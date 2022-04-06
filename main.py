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
#from model.unimodal.gRNA.word2vec_conv import W2V_MK_CNN

from data.data_manager import DataManager
#from training.train import Train
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

        data_list = ["cas9_kim", "cas9_wang", "cas9_xiang", "cas12a_wt_kim"]
        idx = 0 if int(args.target) == 1 else 1 if int(args.target) >= 2 and int(args.target) <= 4 else 2 if int(args.target) >= 5 and int(args.target) <= 7 else 3 if int(args.target) == 8 else -1
        if idx < 0:
            print(f"target parameter : (1:Kim-Cas9, 2:Wang-wt, 3:Wang-HF1, 4:Wang-esp1, 5:Xiang-D2, 6:Xiang-D8, 7:Xiang-D10, 8:Kim-Cas12a)")
            sys.exit(-1)

        self.cfg = {
            
            'seq_path' : f"%s/%s" % (data_cfg['in_dir'], data_cfg[data_list[idx]]),
            'outdir' : f"{data_cfg['out_dir']}/data{args.target}",
            'earlystop' : int(model_cfg["earlystop"]),
            'batch_size' : int(model_cfg["batch"]),
            'EPOCH' : int(model_cfg["epoch"]),
            'seed' : int(model_cfg["seed"]),
            'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

            'kmer' : int(model_cfg["kmer"]),
            'dna2vec_path' : data_cfg['w2v_model'],
            'rna2_mod' : True if args.model.find('2') else 0,
            'chro_mod' : True if args.model.find('3') else 0,           
        }

        os.makedirs(f"{self.cfg['outdir']}/visualize", exist_ok=True)
        os.makedirs(f"{self.cfg['outdir']}/checkpoints", exist_ok=True)

    def define_hyperparam(self, trial):
        #print("define_hyperparam.")
        params = {
            "dropprob" : trial.suggest_float("dropprob", 3e-1, 4e-1),
            "lr": trial.suggest_float("lr", 1e-2, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
            "swa_lr": trial.suggest_float("swa_lr", 5e-3, 1e-2, log=True),
            #"swa_start": trial.suggest_categorical("swa_start", [5, 25])
        }
        
        return params
    
    # def define_model(self, param):
        
    #     model_dict = dict()
    #     model_dict["model"] = EMBD_MK_CNN(drop_prob=0.3, len = 33,  device = self.device).to(self.device)
    #     #model_dict["model"] = W2V_MK_CNN(param['dropprob'], len = 33).to(self.device)
    #     model_dict["optimizer"] = AdamW(model_dict["model"].parameters(), lr = param["lr"], weight_decay=param["weight_decay"])
    #     model_dict["swa_model"] = optim.swa_utils.AveragedModel(model_dict["model"])
    #     model_dict["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(model_dict["optimizer"], T_max = self.EPOCH)
    #     model_dict["swa_scheduler"] = optim.swa_utils.SWALR(optimizer=model_dict["optimizer"], swa_lr = param["swa_lr"])
    #     model_dict["swa_start"] = 5
      
    #     return model_dict

    
    
    def iteration(self, trial, train_loader, valid_loader):

        define_hyperparam()
        #train_df = dict()
        train_df["optimizer"] = AdamW(model_dict["model"].parameters(), lr = param["lr"], weight_decay=param["weight_decay"])

        # ML = Train(self.static_param, model) 
        # corr = ML.run(loader)

        # return corr

        # Generate the model.
        # model = define_model(trial).to(DEVICE)

        # # Generate the optimizers.
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # # Training of the model.
        # model.train()
        # for epoch in range(EPOCHS):
        #     for batch_idx, (data, target) in enumerate(train_loader):
        #         # Limiting training data for faster epochs.
        #         if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
        #             break

        #         data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        #         optimizer.zero_grad()
        #         output = model(data)
        #         loss = F.nll_loss(output, target)
        #         loss.backward()
        #         optimizer.step()

        #     # Validation of the model.
        #     model.eval()
        #     correct = 0
        #     with torch.no_grad():
        #         for batch_idx, (data, target) in enumerate(valid_loader):
        #             # Limiting validation data.
        #             if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
        #                 break
        #             data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
        #             output = model(data)
        #             # Get the index of the max log-probability.
        #             pred = output.argmax(dim=1, keepdim=True)
        #             correct += pred.eq(target.view_as(pred)).sum().item()

        #     accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        #     return accuracy
    
    def objective_cv(self, trial):
        
        DM = DataManager(self.cfg)
        dataset = DM.read_data()

        fold = KFold(n_split = 10, shuffle=True, random_state=0)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(dataset)))):
            
            train_loader, valid_loader = DM.loader(dataset, train_idx, valid_idx)
            corr = self.iteration(trial, train_loader, valid_loader)
            scores.append(corr)
        
        return np.mean(corr)

    
    # def objective(self, trial):
        
    #     best_val_corr = float('Inf')
    #     with mlflow.start_run():
            
    #         param = self.define_hyperparam(trial)
    #         mlflow.log_params(trial.params)

    #         #Initialize network
    #         model = self.define_model(param)

    #         #Train network
    #         best_val_corr = self.train_model(model)

    #     return best_val_corr

    # def print_results(self, study):

    #     print("\n++++++++++++++++++++++++++++++++++\n")
    #     print("Study statistics: ")
    #     print("  Number of finished trials: ", len(study.trials))

    #     print("Best trial:")
    #     trial = study.best_trial #

    #     print("  Trial number: ", trial.number)
    #     print("  Sp corr. (trial value): ", trial.value)

    #     print("  Params: ")
    #     for key, value in trial.params.items():
    #         print("    {}: {}".format(key, value))

    #     df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
    #     df.to_csv(f'{self.out_dir}/optuna_result.csv', sep='\t', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, help="1:Kim-Cas9, 2:Wang-wt, 3:Wang-HF1, 4:Wang-esp1, 5:Xiang-D2, 6:Xiang-D8, 7:Xiang-D10, 8:Kim-Cas12a", required=True)
    parser.add_argument("--model", type=int, help="gRNA module, add RNAss module : 1, Cromatin information : 2, .more than one input is possible (e.g. 1,2)", default=0)
    parser.add_argument("--cv", type=str, default=3)
    args = parser.parse_args()

    runner = Runner(args)
    #-> validate spearman correlation -> maximization

    study = optuna.create_study(study_name="DACO - gRNA module : parameter optimization", direction="maximize",  pruner=optuna.pruners.HyperbandPruner(max_resource="auto")) #
    mlflow_cb = make_mlflow_callback(runner.out_dir)
    study.optimize(runner.objective_cv, n_trials = 10, timeout = 1200, callbacks=[mlflow_cb]) #OK



    # print("Number of finished trials: {}".format(len(study.trials)))

    # runner.print_results(study)


#     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
# complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
