import pandas as pd

kim_cas12a = "/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Kim-Cas12a/41587_2018_BFnbt4061_MOESM39_ESM.xlsx"

def load_file():
    
    file_path = kim_cas12a
    id = ["HT1_1", "HT1_2", "HT2", "HT3", "HEK_lenti", "HEK_plasmid", "HCT_plasmid", "Kleinstiver16", "Chari17", "Kim16"]
    ilist, xlist, ylist, clist = list(), list(), list(), list()
    
    for idx, val in enumerate(id):
        tdata = pd.read_excel(file_path, sheet_name = idx, header = 1, engine = "openpyxl").dropna(how='all')
        tdata = tdata.loc[:, ~tdata.columns.str.contains('^Unnamed')]

        if idx == 0:
            xlist = [k for k in tdata.iloc[:-1,1]]
            ylist = tdata.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()

        elif idx > 0 and idx < 4:

            xlist += [k for k in tdata.iloc[:-1,1]]
            ylist += tdata.iloc[:-1,-1].apply(lambda x: x/100 if x > 0 else 0).to_list()
            print(f'{id[idx]} : {len(xlist)}')

            if idx == 1:
                ilist += ['HT1'] * len(xlist)
                clist += [0] * len(xlist)
            else:
                ilist += [id[idx]] * len(tdata.iloc[:-1,1])
                clist += [0] * len(tdata.iloc[:-1,1])
        else:
            xlist += [k for k in tdata.iloc[:,3]]
            ylist += tdata.iloc[:,-2].apply(lambda x: x/100 if x > 0 else 0).to_list()
            ilist += [id[idx]] * len(tdata.iloc[:,1])
            clist += [0 if k == 'N.A.' else k for k in tdata.iloc[:,-1]]
            print(f'{id[idx]} : {len(xlist)}')
    
    ret = pd.DataFrame({'I' : ilist, 'X' : xlist, 'Y' : ylist, 'C' : clist})
    return ret
    