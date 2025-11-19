import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import KFold
from utils.utils import get_data
from utils.utils import test_evaluate_model
from utils.utils import MultiFeatureDataset
from utils.utils import create_data_loader
from utils.utils import Fold_Cross_Validation_Results
from models.models import AttentionFusionClassifier

def train_classifier(model, train_loader, test_loader, epochs, learning_rate, device):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs_batch, labels_batch in train_loader:
            inputs_batch = [x.to(device) for x in inputs_batch]
            labels_batch = labels_batch.float().to(device)
            optimizer.zero_grad()
            logits, attn_weights = model(inputs_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        auc, aupr, accuracy, pre, recall, f1, mcc = test_evaluate_model(model, test_loader, device)
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} " \
                  f"| AUC: {auc:.4f} | aupr: {aupr:.4f} | Acc: {accuracy:.4f} | " \
                  f"pre: {pre:.4f} | recall: {recall:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    return model



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_save_files = "pre_trained_model_data/Sample_data"

    M ,C ,E ,P,labels =get_data(data_save_files)
    feature_tensors = [M ,C ,E ,P]

    print("Starting models training...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(labels))
    all_results = []

    for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
        MCEPDTI_model_save_path = f"model_result/MCEPDTI_{fold}.joblib"
        print("\n" + "="*50)
        print(f"Starting Fold {fold+1}/5")
        print("\n" + "="*50)
        train_dataset = MultiFeatureDataset([tensor[train_indices] for tensor in feature_tensors],labels[train_indices])
        test_dataset = MultiFeatureDataset([tensor[test_indices] for tensor in feature_tensors],labels[test_indices])
        train_loader = create_data_loader(train_dataset, batch_size=8192, shuffle=True)
        test_loader = create_data_loader(test_dataset, batch_size=8192, shuffle=False)

        input_dims = [tensor.shape[1] for tensor in feature_tensors]
        model = AttentionFusionClassifier(input_dims=input_dims,projection_dim=512,mlp_dim=320,num_heads=8).to(device)

        print(f"Model created with input dimensions: {input_dims}")
        print("Starting training...")
        model= train_classifier(model,train_loader,test_loader,epochs=100,learning_rate=0.001,device=device)
        torch.save(model.state_dict(), MCEPDTI_model_save_path)

        print("\nEvaluating models on test set...")
        auc, aupr, accuracy,pre,recall, f1, mcc= test_evaluate_model(model, test_loader, device)

        fold_result = {'fold': fold+1,'auc': auc,'aupr': aupr,'accuracy': accuracy,'pre': pre,'recall': recall,'f1': f1,'mcc': mcc,}
        all_results.append(fold_result)
        print(fold_result)

    Fold_Cross_Validation_Results(all_results)

if __name__ == "__main__":
   main()

