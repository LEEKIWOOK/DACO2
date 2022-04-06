import subprocess
import multiprocessing
import itertools
import parmap

import numpy as np
import pandas as pd

from arnie.bpps import bpps
from arnie.mfe import mfe

class RNAstruct:
    def __init__(self, domain):
        self.target = domain

    def run_by_row(self, splited_data):
        
        def seq_rnafold(seq):
            grna = seq
            rrna = seq.replace("T", "U")

            if self.target == 5: #Cas12a 
                seq = "AAUUUCUACUCUUGUAGAU" + rrna
            else:
                seq = rrna + "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU"
        
            structure, fenergy, fe_ens, fq_mfe_ens = mfe(seq)
            bp_matrix = bpps(seq)
            output = subprocess.check_output(['perl','./src/data/bpRNA_m.pl', seq, structure])
            looptype = output.decode("utf-8").strip()
            
            return grna, seq, structure, looptype, bp_matrix, fenergy, fe_ens, fq_mfe_ens
        
        ret = [seq_rnafold(x) for x in splited_data]
        return ret

    
    def run_func(self, data, file_prefix):

        num_cores = multiprocessing.cpu_count()
        
        seq_list = data["gRNA"].tolist()
        splited = np.array_split(seq_list, num_cores)
        splited = [x.tolist() for x in splited]

        result = parmap.map(self.run_by_row, splited, pm_pbar=False, pm_processes=num_cores)
        result = list(itertools.chain(*result))
        result = pd.DataFrame(result, columns = ["gRNA", "sequence", "structure", "looptype", "matrix", "free_energy", "free_energy_of_ensemble", "freq_of_mfe_structre_in_ensemble"])
        
        df = result.loc[:, result.columns != "matrix"]
        df.to_csv(f"%s_rs.tsv" % file_prefix, sep='\t', header=None, index=None)
        np.save(f"%s_rs.npy" % file_prefix, result["matrix"].to_list(), allow_pickle=True)

        return result