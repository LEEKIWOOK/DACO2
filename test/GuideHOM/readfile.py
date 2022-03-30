import pandas as pd

kim_cas9 = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Kim-Cas9/aax9249_table_s1.xlsx"
wang = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Wang/41467_2019_12281_MOESM3_ESM.xlsx"
xiang = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Xiang/41467_2021_23576_MOESM4_ESM.xlsx"
kim_cas12a = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Kim-Cas12a/41587_2018_BFnbt4061_MOESM39_ESM.xlsx"

def load_file(file_name):
    
    if file_name == "Kim-Cas9":
        file_path = kim_cas9
        train = pd.read_excel(file_path, header = 0, sheet_name = 0, engine = "openpyxl")
        test = pd.read_excel(file_path, header = 0, sheet_name = 1, engine = "openpyxl")

        ret = {'X': [k[4:27] for k in train.iloc[:,1]] + [k[4:27] for k in test.iloc[:,0]],
            'Y': train.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list() + test.iloc[:,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
        }
        return pd.DataFrame(ret)
    
    elif file_name == "Wang":
        file_path = wang
        data = pd.read_excel(file_path, header = 1, engine = "openpyxl")
        data = data[["21mer", "Wt_Efficiency", "SpCas9-HF1_Efficiency", "eSpCas 9_Efficiency"]].dropna()

        ret = pd.DataFrame({'X' : data["21mer"]+"GG", 'Y_wt' : data.iloc[:,1], 'Y_HF1': data.iloc[:,3], 'Y_esp': data.iloc[:,2]})
        return ret

    elif "Xiang" in file_name:
        file_path = xiang
        id = ["eff_D2","eff_D8","eff_D8_dox","eff_D10", "eff_D10_dox"]
        target_st = 128

        info_list, xlist, ylist, offlist = list(), list(), list(), list()
        oligoseq = pd.read_excel(file_path, sheet_name = 0, header = 0, engine = "openpyxl")

        for idx in range(2,7):
            tdata = pd.read_excel(file_path, sheet_name = idx, header = 0, engine = "openpyxl")

            info_list += [id[idx-2]] * len(tdata.iloc[:,1])
            xlist += [oligoseq.iloc[id-1, 2][target_st:target_st+23] for id in tdata.iloc[:,0]]
            ylist += tdata.iloc[:,3].apply(lambda x: x/100 if x > 0 else 0).to_list()
            offlist += tdata.iloc[:,4].apply(lambda x: x/100 if x > 0 else 0).to_list()
        
        ret = pd.DataFrame({'I' : info_list, 'X' : xlist, 'Y' : ylist, 'O' : offlist})
        return ret
    
    elif "Kim-Cas12a" in file_name:
        file_path = kim_cas12a
        id = ["HT1_1", "HT1_2", "HT2", "HT3", "HEK_lenti", "HEK_plasmid", "HCT_plasmid", "Kleinstiver16", "Chari17", "Kim16"]
        ilist, xlist, ylist, clist = list(), list(), list(), list()
        
        for idx, val in enumerate(id):
            tdata = pd.read_excel(file_path, sheet_name = idx, header = 1, engine = "openpyxl")

            if idx == 0:
                xlist = [k[4:28] for k in tdata.iloc[:-2,1]]
                ylist = tdata.iloc[:-2,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
            elif idx > 0 and idx < 4:

                t = [k[4:28] for k in tdata.iloc[:-2,1]]
                xlist += t
                ylist += tdata.iloc[:-2,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
                if idx == 1:
                    ilist += ['HT1'] * len(xlist)
                    clist += [0] * len(xlist)
                else:
                    ilist += [id[idx]] * len(t)
                    clist += [0] * len(t)
            else:
                t = [k[4:28] for k in tdata.iloc[:,3]]
                xlist += t
                ylist += tdata.iloc[:,-2].apply(lambda x: x/100 if x > 0 else 0).to_list()
                ilist += [id[idx]] * len(t)
                clist += tdata.iloc[:,-1].to_list()
       
        ret = pd.DataFrame({'I' : ilist, 'X' : xlist, 'Y' : ylist, 'C' : clist})
        return ret