import os
import esm
import pickle
from datapreprocess.drug_chemberta import drug_data_chemberta
from datapreprocess.protein_esm import protein_data_esm
from datapreprocess.protein_prot_t5 import protein_data_prot
from datapreprocess.drug_protein_label import drug_protein_label
from datapreprocess.drug_mol2vec import drug_mol2vec


def save_pkl(list,files):
    with open(files, 'wb') as f:
        pickle.dump(list, f)
    print(f"saved to {files}....")

def label(data_files,save_files):
    drug_protein_label_list = drug_protein_label(data_files)
    save_pkl(drug_protein_label_list,files=save_files)


def chemberta(chemberta_model_path,chemberta_merges_path,data_files,save_files):
    drug_chemberta_list = drug_data_chemberta(data_files, chemberta_model_path, chemberta_merges_path)
    save_pkl(drug_chemberta_list,files=save_files)

def mol2vec(data_files,model_files,save_files):
    drug_mol2vec_list = drug_mol2vec(data_files,model_files)
    save_pkl(drug_mol2vec_list,files=save_files)

def esm_2(model_dict,repr_layers,data_files,save_files):
    protein_esm_list = protein_data_esm(data_files,model_dict,repr_layers)
    save_pkl(protein_esm_list,files=save_files)

def prot_t5(data_files,model_path,save_files):
    protein_prot_list = protein_data_prot(data_files,model_path)
    save_pkl(protein_prot_list,files=save_files)


if __name__ == '__main__':
    """
    pre-trained models download:
    esm-2 :  pip install fair-esm  
    mol2vec : https://github.com/samoturk/mol2vec_notebooks/tree/master/Notebooks
    ChemBERTa-77M-MLM : https://huggingface.co/DeepChem/ChemBERTa-77M-MLM/tree/main
    ProtT5-XL-UniRef50 : https://github.com/agemagician/ProtTrans
    """
    print(esm.pretrained.__dict__.keys())

    data_save_files = "pre_trained_model_data/Sample_data"
    data_files = "dataset/Sample_data.csv"

    if not os.path.exists(data_save_files):
        os.makedirs(data_save_files)


    #******   label *******
    label_data_save_path = f"{data_save_files}/drug_protein_label.pkl"
    label(data_files,label_data_save_path)

    # ****** mol2vec *******
    mol2vec_model_path = "Pre_trained_model/model_300dim.pkl"
    mol2vec_data_save_path = f"{data_save_files}/drug_mol2vec.pkl"
    mol2vec(data_files, mol2vec_model_path, mol2vec_data_save_path)


    # ****** chemberta ******
    chemberta_model_path  = "Pre_trained_model/ChemBERTa-77M-MLM"
    chemberta_merges_path = "Pre_trained_model/ChemBERTa-77M-MLM/merges.txt"
    chemberta_save_path = f"{data_save_files}/drug_ChemBERTa.pkl"
    chemberta(chemberta_model_path, chemberta_merges_path, data_files, chemberta_save_path)

    # ****** esm_2 *******
    """
    //esm2_t6_8M_UR50D //esm2_t12_35M_UR50 //esm2_t30_150M_UR50D //esm2_t33_650M_UR50D
    //esm2_t36_3B_UR50D //esm2_t48_15B_UR50D
    print(esm.pretrained.__dict__.keys())
    """
    model_dict = {"models": esm.pretrained.esm2_t33_650M_UR50D}
    repr_layers = 33
    esm_2_save_path = f"{data_save_files}/protein_esm_2.pkl"
    esm_2(model_dict, repr_layers, data_files, esm_2_save_path)

    # ****** ProtT5-XL-UniRef50 *******
    prot_t5_model_path="Pre_trained_model\prot_t5"
    prot_t5_save_path = f"{data_save_files}/protein_prot_t5.pkl"
    prot_t5(data_files,prot_t5_model_path,  prot_t5_save_path)