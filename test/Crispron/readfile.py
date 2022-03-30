import pandas as pd
import argparse


def write_fasta(data, dir):
    
    with open(f'{dir}/30mers.fa', 'w') as f:
        for i in range(len(data['Y'])):
            f.write('>'+data['idx'][i] + '\n' + data['s30'][i])
            if i+1 < len(data['Y']):
                f.write('\n')

    with open(f'{dir}/23mers.fa', 'w') as f:
        for i in range(len(data['Y'])):
            f.write('>'+data['idx'][i] + '\n' + data['s23'][i])    
            if i+1 < len(data['Y']):
                f.write('\n')
    
    out = pd.DataFrame(data)
    out.to_csv(f'{dir}/data.tsv', index=False, sep='\t', header=True)
        

def save_file(args):

    kim_cas9 = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Kim-Cas9"
    xiang = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Xiang"
    
    if args.data == "Kim-Cas9":
        file_path = f'{kim_cas9}/aax9249_table_s1.xlsx'
        train = pd.read_excel(file_path, header = 0, sheet_name = 0, engine = "openpyxl")
        test = pd.read_excel(file_path, header = 0, sheet_name = 1, engine = "openpyxl")

        ret = {
            's30' : [k for k in train.iloc[:,1]] + [k for k in test.iloc[:,0]],
            's23' : [k[4:27] for k in train.iloc[:,1]] + [k[4:27] for k in test.iloc[:,0]],
            'Y' : train.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + test.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        }
        idx = [f'idx_{k}' for k in range(len(ret['s30']))]
        ret['idx'] = idx

        write_fasta(ret, kim_cas9)

    elif args.data == "Xiang":
        file_path = f'{xiang}/41467_2021_23576_MOESM4_ESM.xlsx'

        id = ["eff_D2","eff_D8","eff_D8_dox","eff_D10", "eff_D10_dox"]
        target_st = 128

        info_list, s23list, s30list, ylist, offlist = list(), list(), list(), list(), list()
        oligoseq = pd.read_excel(file_path, sheet_name = 0, header = 0, engine = "openpyxl")

        for idx in range(2,7):
            tdata = pd.read_excel(file_path, sheet_name = idx, header = 0, engine = "openpyxl")

            info_list += [id[idx-2]] * len(tdata.iloc[:,1])

            s23list += [oligoseq.iloc[id-1, 2][target_st:target_st+23] for id in tdata.iloc[:,0]]
            s30list += [oligoseq.iloc[id-1, 2][target_st-4:target_st+23+3] for id in tdata.iloc[:,0]]

            ylist += tdata.iloc[:,3].apply(lambda x: x/100 if x > 0 else 0).to_list()
            offlist += tdata.iloc[:,4].apply(lambda x: x/100 if x > 0 else 0).to_list()
        
        ret = {
            's30' : s30list,
            's23' : s23list,
            'Y' : ylist,
            'idx' : [f'idx_{k}' for k in range(len(s23list))],
            'O' : offlist,
        }

        write_fasta(ret, xiang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Kim-Cas9 / Xiang")
    
    args = parser.parse_args()
    save_file(args)
    