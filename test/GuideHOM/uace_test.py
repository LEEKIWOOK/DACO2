import pandas as pd
import argparse
import os

from Actor import *
from readfile import load_file

out_dir = f"/home/kwlee/Projects_gflas/DACO2/output/GuideHOM"

available_models = {
    'Cas9-DeepHF-wildtype-data': {
        "capsnets": {
            "CNN MSE": ["DeepHF_wt/CNN-u59.ptch"],
            #"RNN MSE": ["DeepHF_wt/RNN-u59.ptch"],
        }, 
        "pca": {
            "CNN MSE": "DeepHF_wt/CNN-u59.pkl",
            #"RNN MSE": "DeepHF_wt/RNN-u59.pkl",
        }, 
        "pam": "[ATGC]GG", "before": False, "use_pam": True,
        "cut_at_start": 0, "cut_at_end": 2, "cuda": torch.cuda.is_available(),
        "guide_length": 20
    },
    'Cas9-DeepHF-eSpCas9-data': {
        "capsnets": {
            #"RNN MSE": ["DeepHF_eSpCas9/RNN-u59.ptch"],
            "CNN MSE": ["DeepHF_eSpCas9/CNN-u59.ptch"],
        }, 
        "pca": {
            #"RNN MSE": "DeepHF_eSpCas9/RNN-u59.pkl",
            "CNN MSE": "DeepHF_eSpCas9/CNN-u59.pkl",
        },
        "pam": "[ATGC]GG", "before": False, "use_pam": True,
        "cut_at_start": 0, "cut_at_end": 2, "cuda": torch.cuda.is_available(),
        "guide_length": 20
    },
    'Cas9-DeepHF-SpCas9HF1-data': {
        "capsnets": {
            #"RNN MSE": ["DeepHF_SpCas9HF1/RNN-u59.ptch"],
            "CNN MSE": ["DeepHF_SpCas9HF1/CNN-u59.ptch"],
        }, 
        "pca": {
            #"RNN MSE": "DeepHF_SpCas9HF1/RNN-u59.pkl",
            "CNN MSE": "DeepHF_SpCas9HF1/CNN-u59.pkl"

        }, 
        "pam": "[ATGC]GG", "before": False, "use_pam": True,
        "cut_at_start": 0, "cut_at_end": 2, "cuda": torch.cuda.is_available(),
        "guide_length": 20
    },
    'Cas12a-DeepCpf1-data': {
        "capsnets": {
            "CNN ELBO": ["DeepCpf1/CNN59.ptch"],
            "CNN MSE": ["DeepCpf1/CNN-u59.ptch"],
        }, 
        "pca": {
            "CNN ELBO": "DeepCpf1/CNN59.pkl",
            "CNN MSE": "DeepCpf1/CNN-u59.pkl",
        },
        "pam": "TTT[ATGC]", "before": True, "use_pam": True,
        "cut_at_start": 0, "cut_at_end": 0, "cuda": torch.cuda.is_available(),
        "guide_length": 20
    }
}

def run_model(data, file):
    
    dir = f'{out_dir}/{file}'
    os.makedirs(out_dir, exist_ok=True)

    for m in available_models.keys():
        if "Cas12a" in m:
            continue

        for mf in available_models[m]['capsnets'].keys():
            model_path = available_models[m]["capsnets"][mf][0]
            model_name = os.path.split(model_path)[0]

            fm = mf.replace(' ','_')
            os.makedirs(f'{dir}/{model_name}/{fm}', exist_ok = True)

            actor = Actor(
                available_models[m]["capsnets"][mf], 
                available_models[m]["pca"][mf],
                available_models[m]["pam"], 
                available_models[m]["use_pam"], 
                available_models[m]["before"], 
                available_models[m]["guide_length"], 
                available_models[m]["cuda"], 
                available_models[m]["cut_at_start"], 
                available_models[m]["cut_at_end"]
            )

            out = pd.DataFrame(actor.on(data))
            #out = pd.merge(data, out, how='outer', left_on='X', right_on='guides_and_pams')
            if len(data['X'] == len(out['guides_and_pams'])):
                ret = out[['activities', 'variances', 'PCA1', 'PCA2']]
                ret = pd.concat([data, ret], axis=1)

            ret['Test_file'] = file
            ret['Trained_data'] = model_name
            ret['Model_arch'] = fm

            ret.to_csv(f'{dir}/{model_name}/{fm}/{model_name}_{fm}_ontarget.tsv', sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Kim-Cas9 / Wang / Xiang / Kim-Cas12a")
    
    args = parser.parse_args()

    df = load_file(args.file)
    run_model(df, args.file)