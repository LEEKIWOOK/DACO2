import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from scipy.stats import spearmanr
from model.unimodal.gRNA.embd_conv import EMBD_MK_CNN

class DataWrapper:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        res = dict()
        for col in self.data.keys():
            if col == "Xg":
                res[col] = torch.tensor(self.data[col][idx], dtype=torch.long)
            else:
                res[col] = torch.tensor(self.data[col][idx], dtype=torch.float)

        return res

class DataManager:
    def __init__(self, batch_size, file):
    
        self.batch_size = batch_size
        self.file = file

    def data_loader(self, data):

        loader = DataLoader(
            DataWrapper(data),
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
        )
        return loader

    def load_file(self):
        def embd_table(seq):
            l = []
            table_key = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            for i in range(len(seq)):
                key = table_key.get(seq[i], -1)
                if key > -1:
                    l.append(key)
            return l    

        self.data = pd.read_csv(self.file, sep='\t', header = None, names = ['window', 'Yg'])
        dataset = {
            'Xg' : self.data.apply(lambda x: np.array(embd_table(x['window'])).T, axis=1).values,
            'Yg' : self.data['Yg'].values
        }
        self.loader = DM.data_loader(dataset)

    def run(self, model_path):
        
        device = torch.device("cpu")
        mymodel = EMBD_MK_CNN(drop_prob = 0.3, len = 33, device = device).cpu()
        state_dict = torch.load(model_path, map_location='cpu')
        mymodel.load_state_dict(state_dict)

        

        retlist = []
        mymodel.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.loader):

                Xg, Yg = data['Xg'].cpu(), data['Yg'].cpu()
                pred = mymodel(Xg)
                output = pred.detach().numpy().tolist()
                retlist += output
        
        self.data['score'] = retlist

        corr = spearmanr(self.data['Yg'], self.data['score'])[0]
        print(f"Test Spearman correlation = {corr}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="test file name")
    parser.add_argument("--cas", type=str, help="Cas protein", default="Cas9")
    args = parser.parse_args()

    if args.cas == "Cas9" or args.cas == "cas9":
        model_path = f"../test/saved_model/best_model.pth"
    # else:
    #     model_path = f"{args.path}/cas12a_daco.pth"

    DM = DataManager(batch_size=64, file = args.file)
    DM.load_file()
    DM.run(model_path)