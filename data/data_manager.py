import os
import random
import numpy as np
import pandas as pd
from functools import reduce

import torch
from torch.utils.data import DataLoader
from data.gRNA.multi_k_model import MultiKModel
#from data.RNAss.rnass_embd import preprocess_inputs
#from utils.torch_util import ForeverDataIterator

DEBUG=False

class DataManager:
    def __init__(self, param):
        self.out_dir = param['out_dir']
        self.batch_size = param['batch_size']
        self.seed = param['seed']        
        self.gRNA_func = self.RNAss_func = self.chro_func = False
        
        if '1' in param['model_run']: #gRNA model -> regression
            self.seq_prefix = param['seq_prefix']
            
            #self.kmer = param['kmer']
            #self.DNA2Vec = MultiKModel(param['dna2vec_path'])
            self.gRNA_func = True
        
        if '2' in param['model_run']: #RNAss model -> regression
            self.rna_prefix = param['rna_prefix']
            #self.rna2st = RNAstruct(param['target_domain'])
            self.RNAss_func = True

        if '3' in param['model_run']: #Chromatin accessibility model -> classifier
            self.chro_prefix = param['chro_prefix']
            #self.chromatin = ChromatinAceess(param['target_domain'])
            self.chro_func = True
  
    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def data_load(self):
        
        func_list = []
        if self.gRNA_func == True:
            data_grna = pd.read_csv(f"%s_annot.tsv" % self.seq_prefix, sep='\t')
                        
            #1.word-to-vec
            #data_grna['window'] = data_grna.apply(lambda x: np.array(self.k_mer_stride(x['window'], self.kmer, 1)).T,axis=1)
            
            #2.one-hot encoding
            #data_grna['window'] = data_grna.apply(lambda x: np.array(self.one_hot(x['window'], 1)).T, axis=1)

            #3.embedding table
            data_grna['window'] = data_grna.apply(lambda x: np.array(self.embd_table(x['window'])).T, axis=1)

            data_grna['efficiency'] = data_grna.apply(lambda x: (x['efficiency'] - min(data_grna['efficiency'])) / (max(data_grna['efficiency']) - min(data_grna['efficiency'])), axis=1)
            func_list.append(data_grna)
            print("grna data loaded.")
                #data_grna = pd.read_csv("/home/kwlee/Projects_gflas/DACO2/data/input/Annot/test_annot.tsv", sep='\t')
                # da = pd.read_csv("/home/kwlee/Projects_gflas/Team_BI/Projects/1.Knockout_project/Data/Finalsets/Backup/Kim2018_Cas12a.feature.csv", sep='\t', usecols=['sgRNA-chrom','sgRNA-pos','sgRNA-seq','Target-gene','Target-transcript','Target-protein','Target-transcript_type'])
                # df = pd.merge(data_grna, da, how='left', left_on='gRNA', right_on='sgRNA-seq')
                # df['sgRNA-pos'] = df['sgRNA-pos'].astype('Int64')
                # df = df.replace(np.nan,'-',regex=True)
                # df = pd.DataFrame(df, columns = ["sgRNA-chrom", "sgRNA-pos", "gRNA", "window", "efficiency", "Target-gene", "Target-transcript", "Target-protein", "Target-transcript_type"])
                # df.to_csv(f"/home/kwlee/Projects_gflas/Team_BI/Projects/DACO2/data/input/Annot/%s_annot.tsv" % "Cas12a_wt_kim", sep='\t', index=None)
        if self.RNAss_func == True:
            # data_rs = self.rna2st.run_func(data_grna, self.rna_prefix)
            data_rs = pd.read_csv(f"%s_rs.tsv" % self.rna_prefix, header=None, names=["gRNA", "sequence", "structure", "looptype", "matrix", "free_energy", "free_energy_of_ensemble", "freq_of_mfe_structre_in_ensemble"], sep='\t')
            mat = np.load(f"%s_rs.npy" % self.rna_prefix)
            data_rs['matrix'] = pd.Series([x for x in mat])
            func_list.append(data_rs)
            print("rnamat data loaded.")
        
        if self.chro_func == True:
            #data_chr = self.chromatin.run_func(data_grna, self.chro_prefix)
            data_chr = pd.read_csv(f"%s_rs.tsv" % self.rna_prefix, header=None, names=["gRNA", "sequence", "structure", "looptype", "matrix", "free_energy", "free_energy_of_ensemble", "freq_of_mfe_structre_in_ensemble"], sep='\t')
            func_list.append(data_chr)
        
        if len(func_list) > 1:
            data = reduce(lambda l, r: pd.merge(l, r, on='gRNA', how='left'), func_list)
        else:
            data = func_list[0]
        dataset = self.split_set(data)

        # train = self.data_loader(train)
        # valid = self.data_loader(valid)
        # test = self.data_loader(test)
        
        #for x in ['train','valid','test']:
        #    self.save_dict(dataset[x], f"{self.out_dir}/{x}.tsv")

        dataloader = self.data_loader(dataset)      
        return dataloader
 
    def embd_table(self, seq):
        l = []
        table_key = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i in range(len(seq)):
            key = table_key.get(seq[i], -1)
            if key > -1:
                l.append(key)
        return l    
    
    def one_hot(self, seq, s):
        l = []
        table_key = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoding_window = s
        for i in range(len(seq)):
            t = seq[i:i + encoding_window]
            one_hot_char = [0] * len(table_key)
            one_hot_idx = table_key.get(t, -1)
            if one_hot_idx > -1:
                one_hot_char[one_hot_idx] = 1
            l.append(one_hot_char)
        return l
    
    def k_mer_stride(self, seq, k, s):
        l = []
        j = 0
        for i in range(len(seq)):
            t = seq[j:j + k]
            if (len(t)) == k:
                vec = self.DNA2Vec.vector(t)
                l.append(vec)
            j += s
        return l
    
    def split_set(self, data, ratio=1.0):
        
        self.seed_everything()
        
        data_size = len(data["gRNA"])
        indice = list(range(data_size))
        np.random.shuffle(indice)

        test_ratio = 0.15
        val_ratio = test_ratio

        test_size = int(np.floor(data_size * test_ratio))
        tv_size = int(np.floor(data_size * (1 - test_ratio) * ratio))
        train_size = int(np.floor(tv_size * (1 - val_ratio)))
        valid_size = int(np.floor(tv_size * val_ratio))

        indices = dict()
        indices["valid"] = random.sample(indice[:valid_size], valid_size)
        indices["test"] = random.sample(
            indice[valid_size : valid_size + test_size], test_size
        )
        indices["train"] = random.sample(
            indice[valid_size + test_size : valid_size + test_size + train_size],
            train_size,
        )

        if self.gRNA_func == True:
            Xg = np.array(data['window'].values.tolist())
            #Xg = [pd.get_dummies(y) for y in [list(x) for x in data['window']]]
            #[list(x) for x in data['window']]
            Yg = data['efficiency'].values           
        # if self.RNAss_func == True:
        #     Xr = np.array(data['window'].values.tolist()) #
        # if self.chro_func == True:
        #     Xc = np.array(data['window'].values.tolist()) #
        
        #rnass = preprocess_inputs(data)
        #efficiency = data['efficiency'].values
        #mat = np.array(data['matrix'].values.tolist())

        def extract(idx):
            dt = {
                'Xg' : [Xg[i] for i in idx],
                #'rnass' : [rnass[i] for i in idx],
                #'rnamat' : [mat[i] for i in idx],
                'Yg' : [Yg[i] for i in idx]
            }
            return dt

        dataset = {
            'train' : extract(indices["train"]),
            'valid' : extract(indices["valid"]),
            'test' : extract(indices["test"])
        }

        return dataset

    def data_loader(self, dataset):

        loaderset = {
            'train' : DataLoader(DataWrapper(dataset["train"]), batch_size = self.batch_size, num_workers=8, drop_last=True),
            'valid' : DataLoader(DataWrapper(dataset["valid"]), batch_size = self.batch_size, num_workers=8, drop_last=True),
            'test' : DataLoader(DataWrapper(dataset["test"]), batch_size = self.batch_size, num_workers=8, drop_last=True)
        }
        return loaderset
        
    def save_dict(self, data, filename):
        data = pd.DataFrame.from_dict(data)
        data.to_csv(filename, index=False, header=True)    
    
class DataWrapper:
    def __init__(self, data, transform=None):
        self.data = data
        #self.Xg = data['Xg']
        #self.Yg = data['Yg']
        # self.dnaseq = data['window']
        # self.rnass = data['rnass']
        # self.rnamat = data['rnamat']
        # self.efficiency = data['efficiency']
        self.transform = transform

    def __len__(self):
        return len(self.data['Xg'])

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
