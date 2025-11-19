from tqdm import tqdm
import pandas as pd
import torch
import re
from collections import Counter
from transformers import T5EncoderModel, T5Tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def prot_bert(sequence_examples,model,tokenizer):
    n = len(sequence_examples)
    sequence_examples = [sequence_examples]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb = embedding_repr.last_hidden_state[:,:n,:]
    return emb.squeeze(0).mean(dim=0)
def protein_data_prot(file,model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    data =pd.read_csv(file)
    peptide_list = data['TargetSequence'].tolist()
    unique_embeddings = {}
    protein_prot_list = []
    unique_peptides = list(Counter(peptide_list).keys())
    print(f"Number of unique peptide segments: {len(unique_peptides)}")
    for peptides in tqdm(unique_peptides, desc="Processing sequence prot..."):
        if len(peptides)<=4000:
            unique_embeddings[peptides] = prot_bert(peptides,model,tokenizer)
            torch.cuda.empty_cache()
        elif len(peptides)>4000:
            print(peptides)
            split = len(peptides)//2
            peptides1 = peptides[:split]
            peptides2 = peptides[split:]
            A  = prot_bert(peptides1,model,tokenizer)
            B  = prot_bert(peptides2,model,tokenizer)
            unique_embeddings[peptides] = (A+B)/2
    for sequence in tqdm(peptide_list):
        prot_embeddings = unique_embeddings[sequence]
        protein_prot_list.append(prot_embeddings)
    return protein_prot_list