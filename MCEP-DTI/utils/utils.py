import pickle
import torch
import numpy as np
from sklearn.metrics import roc_curve,precision_score, recall_score,\
      f1_score, matthews_corrcoef,roc_auc_score, average_precision_score, accuracy_score

class MultiFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_tensors, labels):
        self.feature_tensors = feature_tensors
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = [ft[idx] for ft in self.feature_tensors]
        label = self.labels[idx]
        return features, label

def collate_fn(batch):
    num_modalities = len(batch[0][0])
    features_by_modality = [[] for _ in range(num_modalities)]
    labels = []
    for sample in batch:
        features, label = sample
        for i in range(num_modalities):
            features_by_modality[i].append(features[i])
        labels.append(label)
    stacked_features = [torch.stack(modality_features) for modality_features in features_by_modality]
    stacked_labels = torch.stack(labels)
    return stacked_features, stacked_labels

def create_data_loader(dataset, batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def get_data(save_files):
    drug_mol2vec_list = load_data(f"{save_files}/drug_mol2vec.pkl")
    drug_chemberta_list = load_data(f"{save_files}/drug_ChemBERTa.pkl")
    protein_esm_list = load_data(f"{save_files}/protein_esm_2.pkl")
    protein_prot_t5_list = load_data(f"{save_files}/protein_prot_t5.pkl")
    drug_protein_label_list = load_data(f"{save_files}/drug_protein_label.pkl")

    drug_mol2vec = torch.stack(drug_mol2vec_list).float()
    drug_chemberta = torch.stack(drug_chemberta_list).float()
    protein_esm = torch.stack(protein_esm_list).float()
    protein_prot_t5 = torch.stack(protein_prot_t5_list).float()
    labels = torch.tensor(drug_protein_label_list, dtype=torch.float32)
    return drug_mol2vec, drug_chemberta, protein_esm, protein_prot_t5, labels

def test_evaluate_model(model,val_loader,device):
    def find_best_threshold_ROC(all_labels, all_probs):
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        J = tpr - fpr
        best_threshold_index = np.argmax(J)
        best_threshold = thresholds[best_threshold_index]
        return best_threshold
    model.eval()
    all_probs = []
    all_labels = []
    fused_list = []
    with torch.no_grad():
        for inputs_batch, labels_batch in val_loader:
            inputs_batch = [x.to(device) for x in inputs_batch]
            labels_batch = labels_batch.cpu().numpy()
            logits, fused= model(inputs_batch)
            fused_list.extend(fused)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels_batch)
    threshold = find_best_threshold_ROC(all_labels, all_probs)
    y_pred = (all_probs > threshold).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, y_pred)
    precision = precision_score(all_labels, y_pred)
    recall = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)
    mcc = matthews_corrcoef(all_labels, y_pred)
    return auc,aupr,accuracy,precision,recall,f1,mcc


def Fold_Cross_Validation_Results(all_results):
    avg_auc = np.mean([r['auc'] for r in all_results])
    avg_aupr = np.mean([r['aupr'] for r in all_results])
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_pre = np.mean([r['pre'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_mcc = np.mean([r['mcc'] for r in all_results])

    print("\n" + "=" * 50)
    print("5-Fold Cross Validation Results:")
    print("=" * 50)
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Average AUPR: {avg_aupr:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average pre: {avg_pre:.4f}")
    print(f"Average recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average MCC: {avg_mcc:.4f}")

    print("\nDetailed Results per Fold:")
    for result in all_results:
        print(f"Fold {result['fold']}: AUC={result['auc']:.4f}, AUPR={result['aupr']:.4f}, "
              f"Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, MCC={result['mcc']:.4f}")
    print("Training completed.")
