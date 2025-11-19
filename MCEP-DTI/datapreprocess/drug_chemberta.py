from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import math
from collections import Counter

def chemBERTa(smiles,tokenizer,chemBERTa_model):
    inputs = tokenizer(smiles, return_tensors='pt')
    with torch.no_grad():
        outputs = chemBERTa_model(**inputs)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1].squeeze(0)
    vector = last_hidden_state[1:-1, :]
    avg_vector = torch.mean(vector, dim=0)
    return avg_vector

def drug_data_chemberta(file,model_path,merges_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, merges_file=merges_file)
    chemBERTa_model = AutoModelForMaskedLM.from_pretrained(model_path, output_hidden_states=True)
    data =pd.read_csv(file)
    smiles_list = data['SMILES'].tolist()
    unique_smiles_list = list(Counter(smiles_list).keys())
    print(f"Number of unique smiles: {len(unique_smiles_list)}")
    unique_embeddings = {}
    drug_chemberta_list = []
    for smiles in tqdm(unique_smiles_list, desc="Processing SMILES Graphs"):
        if len(smiles) > 512:
            print(smiles)
            n_parts = math.ceil(len(smiles) / 512)
            part_length = len(smiles) // n_parts
            if len(smiles) % n_parts != 0:
                part_length += 1
            smiles_parts = []
            for i in range(n_parts):
                start_idx = i * part_length
                end_idx = min((i + 1) * part_length, len(smiles))
                smiles_parts.append(smiles[start_idx:end_idx])
            embeddings = []
            for part in smiles_parts:
                b = chemBERTa(part, tokenizer, chemBERTa_model)
                embeddings.append(b)
            B = sum(embeddings) / len(embeddings)

            has_nan = torch.isnan(B).any()

            if has_nan :
                print(f"Does the vector contain NaN: {has_nan}")
                print("Please remove smiles:",smiles)

            unique_embeddings[smiles] = B
        else:
            A = chemBERTa(smiles, tokenizer, chemBERTa_model)

            has_nan = torch.isnan(A).any()

            if has_nan :
                print(f"Does the vector contain NaN: {has_nan}")
                print("Please remove smiles:",smiles)

            unique_embeddings[smiles] = A

    for smiles in tqdm(smiles_list):
        drug_chemberta_embeddings = unique_embeddings[smiles]
        drug_chemberta_list.append(drug_chemberta_embeddings)
    return drug_chemberta_list