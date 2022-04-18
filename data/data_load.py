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

        ret = pd.DataFrame({
            'X': [k[4:27] for k in tr.iloc[:,1]] + [k[4:27] for k in te.iloc[:,0]],
            'X30': tr.iloc[:,1].to_list() + te.iloc[:,0].to_list(),
            'Y': tr.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + te.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })
        
    elif cfg['seqidx'] == 1: #cas9_wang
        data = pd.read_excel(cfg['seqpath'], header = 1, engine = "openpyxl").dropna(how='all')
        data = data[["21mer", "Wt_Efficiency", "SpCas9-HF1_Efficiency", "eSpCas 9_Efficiency"]].dropna()
        yidx = 1 if cfg['target'] == 2 else 3 if cfg['target'] == 3 else 2
        
        ret = pd.DataFrame({
            'X' : data["21mer"]+"GG", 
            'Y' : data.iloc[:,yidx]
        })

    elif cfg['seqidx'] == 2: #cas9_xiang
        oligoseq = pd.read_excel(cfg['seqpath'], sheet_name = 0, header = 0, engine = "openpyxl").dropna(how='all')
        yidx = 2 if cfg['target'] == 5 else 3 if cfg['target'] == 6 else 5
        target_st = 128

        data = pd.read_excel(cfg['seqpath'], sheet_name = yidx, header = 0, engine = "openpyxl").dropna(how='all')

        ret = pd.DataFrame({
            'X': [oligoseq.iloc[id-1, 2][target_st:target_st+23].upper() for id in data.iloc[:,0]],
            'X30': [oligoseq.iloc[id-1, 2][target_st-4:target_st+23+3].upper() for id in data.iloc[:,0]],
            'Y': data.iloc[:,3].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })
    
    elif cfg['seqidx'] == 3: #cas12a_kim
        tr = pd.read_excel(cfg['seqpath'], sheet_name = 0, header = 1, engine = "openpyxl").dropna(how='all')
        te = pd.read_excel(cfg['seqpath'], sheet_name = 1, header = 1, engine = "openpyxl").dropna(how='all')
        tr = tr.loc[:, ~tr.columns.str.contains('^Unnamed')]
        te = te.loc[:, ~te.columns.str.contains('^Unnamed')]

        ret = pd.DataFrame({
            'X': [k[4:28] for k in tr.iloc[:-1,1]] + [k[4:28] for k in te.iloc[:-1,1]],
            'X30': [k for k in tr.iloc[:-1,1]] + [k for k in te.iloc[:-1,1]],
            'Y': tr.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + te.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        })

    if cfg['rna2_mod']:
        #ret = load_mfe(cfg['target'], ret)
        calc_mfe(cfg['seqidx'], ret)

    #elif cfg['chro_mod']:
    #    load_chro(ret)
    
    
    xidx = 'X30' if cfg['seqidx'] != 1 and cfg['seqlen'] == 30 else 'X'
    if cfg['embd'] == 0: #one-hot encoding
        ret['X'] = ret.apply(lambda x: np.array(one_hot(x[xidx], 1)).T, axis=1)

    elif cfg['embd'] == 1: #embd table
        ret['X'] = ret.apply(lambda x: np.array(embd_table(x[xidx])).T, axis=1)

    elif cfg['embd'] == 2: #word-to-vec
        ret['X'] = ret.apply(lambda x: np.array(k_mer_stride(x[xidx], cfg['kmer'], cfg['stride'], cfg['dna2vec_path'])).T,axis=1)
    
    if 'X30' in ret.keys():
        ret = ret.drop(['X30'], axis = 1)
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

        if cfg['rna2_mod']:
            self.mfe = data['R'][tvidx].to_list()
        if cfg['chro_mod']:
            self.chro = data['C'][tvidx].to_list()

        self.transform = transform
        self.xdtype = torch.float if cfg['embd'] == 2 else torch.long


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