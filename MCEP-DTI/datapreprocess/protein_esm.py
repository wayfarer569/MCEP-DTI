from collections import Counter
from tqdm import tqdm
import pandas as pd
import torch
def get_sem_embedings(protein_peptide_list,models_dict,repr_layers):
    unique_peptides = list(Counter(protein_peptide_list).keys())
    print(f"Number of unique peptide segments: {len(unique_peptides)}")
    model, alphabet = models_dict["models"]()
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda()
    model.eval()
    unique_embeddings = {}
    protein_esm_embeddings_list = []
    for idx, protein_peptide in tqdm(enumerate(unique_peptides), desc="Processing sem_embeddings", total=len(unique_peptides), ascii=True):
        raw_batch = [(str(idx), protein_peptide)]
        batch_labels, batch_strs, batch_tokens = batch_converter(raw_batch)
        batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layers])
            embeddings = results["representations"][repr_layers]
            seq_embeddings = embeddings[0, 1:-1, :]
            seq_embeddings = seq_embeddings.mean(dim=0)
            unique_embeddings[protein_peptide] = seq_embeddings
        torch.cuda.empty_cache()
    for protein_peptide in tqdm(protein_peptide_list):
        protein_esm_embeddings = unique_embeddings[protein_peptide]
        protein_esm_embeddings_list.append(protein_esm_embeddings)
    return protein_esm_embeddings_list
def protein_data_esm(file,model_dict,repr_layers):
    data =pd.read_csv(file)
    peptide_list = data['TargetSequence'].tolist()
    protein_esm_list = get_sem_embedings(peptide_list,model_dict,repr_layers= repr_layers)
    return protein_esm_list
