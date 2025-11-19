import torch
import numpy as np
from models.models import AttentionFusionClassifier
import pandas as pd
import esm
from datapreprocess.drug_chemberta import drug_data_chemberta
from datapreprocess.protein_esm import protein_data_esm
from datapreprocess.protein_prot_t5 import protein_data_prot
from datapreprocess.drug_mol2vec import drug_mol2vec
from utils.utils import get_data

def collate_fn(batch):
    num_modalities = len(batch[0])
    features_by_modality = [[] for _ in range(num_modalities)]
    for sample in batch:
        features = sample
        for i in range(num_modalities):
            features_by_modality[i].append(features[i])
    stacked_features = [torch.stack(modality_features) for modality_features in features_by_modality]
    return stacked_features

def create_data_loader(dataset, batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

class MultiFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_tensors):
        self.feature_tensors = feature_tensors
    def __len__(self):
        return len(self.feature_tensors[0])
    def __getitem__(self, idx):
        features = [ft[idx] for ft in self.feature_tensors]
        return features


def evaluate_model(model,test_loader,device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs_batch in test_loader:
            inputs_batch = [x.to(device) for x in inputs_batch]
            logits, _ = model(inputs_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            if np.ndim(probs) == 0:
                all_probs.append(probs)
            else:
                all_probs.extend(probs)
    return all_probs

def get_preprocess_data(data_files):

    mol2vec_model_path = "Pre_trained_model/model_300dim.pkl"
    drug_mol2vec_list = drug_mol2vec(data_files, mol2vec_model_path)

    chemberta_model_path  = "Pre_trained_model/ChemBERTa-77M-MLM"
    chemberta_merges_path = "Pre_trained_model/ChemBERTa-77M-MLM/merges.txt"
    drug_chemberta_list = drug_data_chemberta(data_files, chemberta_model_path, chemberta_merges_path)

    model_dict = {"models": esm.pretrained.esm2_t33_650M_UR50D}
    repr_layers = 33
    protein_esm_list = protein_data_esm(data_files, model_dict, repr_layers)

    prot_t5_model_path = "Pre_trained_model\prot_t5"
    protein_pro_tt5_list = protein_data_prot(data_files, prot_t5_model_path)

    drug_mol2vec_data     = torch.stack(drug_mol2vec_list).float()
    drug_chemberta_data   = torch.stack(drug_chemberta_list).float()
    protein_esm_data     = torch.stack(protein_esm_list).float()
    protein_prot_t5_data  = torch.stack(protein_pro_tt5_list).float()
    return drug_mol2vec_data ,drug_chemberta_data ,protein_esm_data ,protein_prot_t5_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = "model_result/MCEPDTI_0.joblib"
    data_files =  "dataset/Sample_data.csv"
    #M, C, E, P, _= get_data("pre_trained_model_data/Sample_data")
    M ,C  ,E ,P = get_preprocess_data(data_files)
    feature_tensors = [M ,C ,E,P]
    test_dataset = MultiFeatureDataset(feature_tensors)
    test_loader = create_data_loader(test_dataset, batch_size=8192, shuffle=False)

    input_dims = [tensor.shape[1] for tensor in feature_tensors]
    model = AttentionFusionClassifier(input_dims=input_dims,projection_dim=512,mlp_dim=320,num_heads=8).to(device)
    model.load_state_dict(torch.load(model_file))
    all_probs = evaluate_model(model, test_loader, device)
    df = pd.DataFrame(all_probs, columns=['probability'])
    print(df)
    df.to_csv('predict_result.csv', index=False)

if __name__ == '__main__':
    main()



