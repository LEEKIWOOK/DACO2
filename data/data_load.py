import sys
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.calc_mfe import calc_mfe #,load_mfe 
from utils.general_util import one_hot, embd_table, k_mer_stride
#from utils.torch_util import seed_everything

def data_read(cfg):

    if cfg['seqidx'] == 0: #cas9_kim
        tr = pd.read_excel(cfg['seqpath'], header = 0, sheet_name = 0, engine = "openpyxl").dropna(how='all')
        te = pd.read_excel(cfg['seqpath'], header = 0, sheet_name = 1, engine = "openpyxl").dropna(how='all')
        
        if cfg['seqlen'] == 30:
            x = tr.iloc[:,1].to_list() + te.iloc[:,0].to_list()
        else:
            if cfg['seqlen'] >= 23:
                op = int(math.ceil((cfg['seqlen'] - 23)/2))
                st = 4 - op; ed = st + cfg['seqlen'] 
            else:
                op = 23 - cfg['seqlen'] 
                st = 4 + op; ed = st + cfg['seqlen']
            x = [k[st:ed] for k in tr.iloc[:,1]] + [k[st:ed] for k in te.iloc[:,0]]

        ret = pd.DataFrame({
            
            'X': x,
            'Y': tr.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + te.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })
        
    elif cfg['seqidx'] == 1: #cas9_wang
        data = pd.read_excel(cfg['seqpath'], header = 1, engine = "openpyxl").dropna(how='all')
        data = data[["21mer", "Wt_Efficiency", "SpCas9-HF1_Efficiency", "eSpCas 9_Efficiency"]].dropna()
        yidx = 1 if cfg['target'] == 2 else 3 if cfg['target'] == 3 else 2

        if cfg['seqlen'] == 23:
            x = data["21mer"]+"GG"
        elif cfg['seqlen'] < 23:
            op = 23 - cfg['seqlen'] 
            st = op; ed = st + (cfg['seqlen'] - 2)
            print(f"st:{st}, ed:{ed}")
            x = [k[st:ed]+"GG" for k in data.iloc[:,0]]
        else:
            print(f"system error, wong grna length 23")
            sys.exit(-1)
        
        ret = pd.DataFrame({
            'X' : x,
            'Y' : data.iloc[:,yidx]
        })

    elif cfg['seqidx'] == 2: #cas9_xiang
        oligoseq = pd.read_excel(cfg['seqpath'], sheet_name = 0, header = 0, engine = "openpyxl").dropna(how='all')
        yidx = 2 if cfg['target'] == 5 else 3 if cfg['target'] == 6 else 5
        data = pd.read_excel(cfg['seqpath'], sheet_name = yidx, header = 0, engine = "openpyxl").dropna(how='all')
        target_st = 128

        if cfg['seqlen'] == 30:
            st = target_st; ed = target_st + cfg['seqlen']
        elif cfg['seqlen'] >= 23:
            op = int(math.ceil((cfg['seqlen'] - 23)/2))
            st = target_st + (4 - op); ed = st + cfg['seqlen']
        else:
            op = 23 - cfg['seqlen']
            st = target_st + 4 + op; ed = st + cfg['seqlen']
        
        x = [oligoseq.iloc[id-1, 2][st:ed].upper() for id in data.iloc[:,0]]

        ret = pd.DataFrame({
            'X': x,
            'Y': data.iloc[:,3].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })
    
    elif cfg['seqidx'] == 3: #cas12a_kim
        tr = pd.read_excel(cfg['seqpath'], sheet_name = 0, header = 1, engine = "openpyxl").dropna(how='all')
        te = pd.read_excel(cfg['seqpath'], sheet_name = 1, header = 1, engine = "openpyxl").dropna(how='all')
        tr = tr.loc[:, ~tr.columns.str.contains('^Unnamed')]
        te = te.loc[:, ~te.columns.str.contains('^Unnamed')]

        if cfg['seqlen'] == 34: #full error
            x = tr.iloc[:-1,1].to_list() + te.iloc[:-1,1].to_list()
        else:
            if cfg['seqlen'] >= 24:
                op = int(math.ceil((cfg['seqlen'] - 24)/2))
                st = (4 - op); ed = st + cfg['seqlen']
            else:
                op = 24 - cfg['seqlen']
                st = 4; ed = st + cfg['seqlen']
            x = [k[st:ed] for k in tr.iloc[:-1,1]] + [k[st:ed] for k in te.iloc[:-1,1]]

        ret = pd.DataFrame({
            'X': x,
            'Y': tr.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + te.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })

    #if cfg['rna2_mod']:
        #ret = load_mfe(cfg['target'], ret)
        #calc_mfe(cfg['target'], ret)

    #elif cfg['chro_mod']:
    #    load_chro(ret)
    
    
    if cfg['embd'] == 0: #one-hot encoding
        ret['X'] = ret.apply(lambda x: np.array(one_hot(x['X'], 1)).T, axis=1)

    elif cfg['embd'] == 1: #embd table
        ret['X'] = ret.apply(lambda x: np.array(embd_table(x['X'])).T, axis=1)

    elif cfg['embd'] == 2: #word-to-vec
        ret['X'] = k_mer_stride(ret['X'], cfg)

    return ret

def data_loader(cfg, data, tidx, vidx):

    #seed_everything(cfg["seed"])
    tloader = DataLoader(DataWrapper(data, tidx, cfg), batch_size=cfg['batch_size'], num_workers=8, shuffle = True)
    vloader = DataLoader(DataWrapper(data, vidx, cfg), batch_size=cfg['batch_size'], num_workers=8, shuffle = True)

    return tloader, vloader

class DataWrapper:
    def __init__(self, data, tvidx, cfg, transform=None):
        np.random.shuffle(tvidx)
        data = data.reset_index()
        self.mod = (cfg['rna2_mod'], cfg['chro_mod'])
        self.data = data['X'][tvidx].to_list()
        self.target = data['Y'][tvidx].to_list()

        if self.mod[0]:
            self.mfe = data['R'][tvidx].to_list()
        if self.mod[1]:
            self.chro = data['C'][tvidx].to_list()

        self.transform = transform
        self.xdtype = torch.long if cfg['embd'] == 1 else torch.float


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = dict()
        res['X'] = torch.tensor(self.data[idx], dtype = self.xdtype)
        res['Y'] = torch.tensor(self.target[idx], dtype = torch.float)
        
        if self.mod[0]:
            res['R'] = torch.tensor(self.mfe[idx], dtype=torch.float)
        if self.mod[1]:
            res['C'] = torch.tensor(self.chro[idx], dtype=torch.float)

        return res