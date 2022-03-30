import pandas as pd
import argparse
import os
import sys
from functools import reduce

from readfile import load_file

sys.path.append('/home/kwlee/Projects_gflas/DACO2/tools/deepHF/DeepHF')
from prediction_util import get_embedding_data, output_prediction
from feature_util import feature_options

out_dir = f"/home/kwlee/Projects_gflas/DACO2/output/DeepHF"

def run_model(data, file):
    
    dir = f'{out_dir}/{file}'
    os.makedirs(dir, exist_ok=True)

    gRNA = [k[0:21] for k in data['X']]
    PAM = [k[20:23] for k in data['X']]
    Cut_Pos = [17] * len(PAM)
    Strand = ['*'] * len(PAM)

    pd.set_option('Precision', 5)
    df = pd.DataFrame( {'Strand': Strand, 'Cut_Pos': Cut_Pos, '21mer': gRNA, 'PAM': PAM}, columns=['Strand', 'Cut_Pos', '21mer', 'PAM'] )
    X,X_biofeat = get_embedding_data(df,feature_options)

    wt_u6_score = output_prediction( [X,X_biofeat], df, 'wt_u6' )
    wt_u6_score.rename(columns = {'Efficiency':'u6_pred'}, inplace=True)

    wt_t7_score = output_prediction( [X,X_biofeat], df, 'wt_t7' )
    wt_t7_score.rename(columns = {'Efficiency':'t7_pred'}, inplace=True)
    
    esp_score = output_prediction( [X,X_biofeat], df, 'esp' )
    esp_score.rename(columns = {'Efficiency':'esp_pred'}, inplace=True)

    hf_score = output_prediction( [X,X_biofeat], df, 'hf' )
    hf_score.rename(columns = {'Efficiency':'hf_pred'}, inplace=True)


    m_score = reduce(lambda l, r: pd.merge(l, r, on=['X'], how='outer'), [wt_u6_score, wt_t7_score, esp_score, hf_score])
    ret = pd.merge(data, m_score, how='outer', on='X')

    ret['Test_file'] = file
    ret.to_csv(f'{dir}/ontarget.tsv', sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Kim-Cas9 / Wang / Xiang")
    args = parser.parse_args()

    df = load_file(args.file)
    run_model(df, args.file)