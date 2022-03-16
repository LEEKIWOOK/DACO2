import sys
import numpy as np
import subprocess
import pandas as pd
from Bio.Seq import Seq
from arnie.bpps import bpps
from arnie.mfe import mfe

#from preprocess.rna_struct import RNAstruct
#from preprocess.chromatin_access import ChromatinAceess

class Generator:
    def __init__(self, filename): 
        super(Generator, self).__init__()

        self.in_file = f"../data/unlabel/{filename}"
        self.out_file = f"../data/unlabel/{filename}.tsv"
        self.out_mat = f"../data/unlabel/{filename}.npy"

    def rnast(self):
        def run_by_row(row):
            grna = row['gRNA'].replace("T", "U")
            if row['strand'] == '-':
                grna = str(Seq(grna).reverse_complement())
            seq = grna + "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU"
            structure, _, _, _ = mfe(seq)
            bp_matrix = bpps(seq)
            output = subprocess.check_output(['perl','./preprocess/bpRNA_m.pl', seq, structure])
            looptype = output.decode("utf-8").strip()

            output = {
                "seq": seq,
                "structure": structure,
                "looptype": looptype,
                "matrix": bp_matrix
            }

            ret = pd.concat([row, pd.Series(output)])
            return ret
        
        data = pd.read_csv(self.in_file, sep='\t')
        result = data.apply(run_by_row, axis = 1)
        
        df = result.loc[:, result.columns != "matrix"]
        df.to_csv(self.out_file, sep='\t', index=None)
        np.save(self.out_mat, result["matrix"].to_list(), allow_pickle=True)


    #def chromatin(self, data):
    #def save_data(self, data):



if __name__ == "__main__":
    
    G = Generator(sys.argv[1])
    G.rnast()
    #data = G.chromatin(data)
    #G.save_data(data)