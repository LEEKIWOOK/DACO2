import pandas as pd

from readfile import load_file

out_dir = f"/home/kwlee/Projects_gflas/DACO2/output/DeepCpf1"

def run_model(data):
    
    ret = {
        'idx' : [k+1 for k in range(len(data['X']))],
        'Sequence' : data['X'],
        'Chromatin' : [int(k) for k in data['C']]
    }

    df = pd.DataFrame(ret)
    df.to_csv(f'{out_dir}/input.tsv', index=False, sep='\t')
    data.to_csv(f'{out_dir}/original.tsv', index=False, sep='\t')

if __name__ == "__main__":

    df = load_file()
    run_model(df)