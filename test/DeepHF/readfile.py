import pandas as pd
from functools import reduce

kim_cas9 = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Kim-Cas9/aax9249_table_s1.xlsx"
wang = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Wang/41467_2019_12281_MOESM3_ESM.xlsx"
xiang = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Xiang/41467_2021_23576_MOESM4_ESM.xlsx"

def load_file(file_name):
    
    if file_name == "Kim-Cas9":
        file_path = kim_cas9
        train = pd.read_excel(file_path, header = 0, sheet_name = 0, engine = "openpyxl").dropna(how='all')
        test = pd.read_excel(file_path, header = 0, sheet_name = 1, engine = "openpyxl").dropna(how='all')

        ret = {'X': [k[4:27] for k in train.iloc[:,1]] + [k[4:27] for k in test.iloc[:,0]], ##
            'Y': train.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + test.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        }
        return pd.DataFrame(ret)
    
    elif file_name == "Wang":
        file_path = wang
        data = pd.read_excel(file_path, header = 1, engine = "openpyxl").dropna(how='all')
        data = data[["21mer", "Wt_Efficiency", "SpCas9-HF1_Efficiency", "eSpCas 9_Efficiency"]].dropna()

        ret = pd.DataFrame({'X' : data["21mer"]+"GG", 'Y_wt' : data.iloc[:,1], 'Y_HF1': data.iloc[:,3], 'Y_esp': data.iloc[:,2]})
        return ret

    elif "Xiang" in file_name:
        file_path = xiang
        id = ["eff_D2","eff_D8","eff_D8_dox","eff_D10", "eff_D10_dox"]
        target_st = 128

        xlist, ylist, offlist = list(), list(), list()
        ret = list()
        oligoseq = pd.read_excel(file_path, sheet_name = 0, header = 0, engine = "openpyxl").dropna(how='all')

        for idx in range(2,7):
            tdata = pd.read_excel(file_path, sheet_name = idx, header = 0, engine = "openpyxl").dropna(how='all')

            #info_list += [id[idx-2]] * len(tdata.iloc[:,1])
            name = [id[idx-2]]
            xlist = [oligoseq.iloc[id-1, 2][target_st:target_st+23].upper() for id in tdata.iloc[:,0]]
            ylist = tdata.iloc[:,3].apply(lambda x: x/100 if x > 0 else 0).to_list()
            offlist = tdata.iloc[:,4].apply(lambda x: x/100 if x > 0 else 0).to_list()

            td = {'X': xlist, 'Y': ylist, 'O': offlist}
            td[f'{name}_Y'] = td.pop('Y')
            td[f'{name}_O'] = td.pop('O')
            ret.append(pd.DataFrame(td))

        mret = reduce(lambda l, r: pd.merge(l, r, on=['X'], how='outer'), [ret[0], ret[1], ret[2], ret[3], ret[4]])

        return mret
    