import pandas as pd
import numpy as np
from gensim.models import word2vec
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
import torch

def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}
    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element
    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])
    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)
def get_ECFP(mol, radio):
    ECFPs = mol2alt_sentence(mol, radio)
    if len(ECFPs) % (radio + 1) != 0:
        ECFPs = ECFPs[:-(len(ECFPs) % (radio + 1))]
    ECFP_by_radio = list((np.array(ECFPs).reshape((int(len(ECFPs) / (radio + 1)), (radio + 1))))[:, radio])
    return ECFP_by_radio
def get_drug_ecpf(smile,smile2ecfp):
    ECPFs = np.zeros(300, dtype=float)
    ecpfs = get_ECFP(Chem.MolFromSmiles(smile), 1)
    num = 0
    for ecpf in ecpfs:
        try:
            out = smile2ecfp.wv.word_vec(ecpf)
            num+=1
        except Exception as err:
            # print(err, temp_ecpf, temp_num)
            continue
        ECPFs += out
    if num ==0:
        return ECPFs
    else:
        ECPFs = ECPFs/num
        return ECPFs
def drug_mol2vec(file,model_files):
    smile2ecfp = word2vec.Word2Vec.load(model_files)
    data =pd.read_csv(file)
    smiles_list = data['SMILES'].tolist()
    A=[]
    for smiles in tqdm(smiles_list):
        a = get_drug_ecpf(smiles,smile2ecfp)
        a_tensor = torch.tensor(a, dtype=torch.float)
        A.append(a_tensor)
    return A
