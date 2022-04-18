import sys
import pandas as pd
from data.arnie.mfe import mfe

def calc_mfe(target, data):

    if target == 8: #Cas12a
        ret = [ mfe("AAUUUCUACUCUUGUAGAU" + k[0:19].replace("T", "U")) for k in data['X']]
    else: #Cas9
        ret = [mfe(k[0:19].replace("T", "U") + "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU") for k in data['X']]
    data = pd.concat([data['X'], pd.DataFrame(ret)], axis=1)
    data.to_csv(f"{target}.tsv", sep='\t', header=True, index=False)
    sys.exit(0)

# def load_mfe(target, data):

