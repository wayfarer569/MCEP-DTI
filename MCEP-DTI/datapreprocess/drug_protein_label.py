import pandas as pd



def drug_protein_label(file):
    data =pd.read_csv(file)
    label_list = data['Label'].tolist()
    return label_list