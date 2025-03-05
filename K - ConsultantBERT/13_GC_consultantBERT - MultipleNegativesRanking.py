from sentence_transformers import SentenceTransformer, InputExample, models, losses, util
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModel, set_seed
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
import os
import json
import random
from helpers import *

class ConsultantBERT:
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased", max_seq_length: int = 512):
        set_seed(42)  # For reproducibility
        
        # Create bi-encoder architecture as described in paper
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True  # Use mean pooling as per paper
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
    def prepare_training_data(self, pos_pairs: List, neg_pairs: List, batch_size: int = 4):
        """Prepare training examples from positive and negative pairs"""
        train_examples = []
        
        # Add positive pairs
        for pair in pos_pairs:
            # Take first 512 tokens of each document as per paper
            train_examples.append(InputExample(
                texts=[pair[0][:512], pair[1][:512]], 
                label=1.0
            ))
            
        # Add negative pairs
        for pair in neg_pairs:
            train_examples.append(InputExample(
                texts=[pair[0][:512], pair[1][:512]], 
                label=0.0
            ))
                
        return DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,  # As per paper
                   num_epochs: int = 5,  # As per paper
                   fold_size: int = 10, 
                   lr: float = 2e-5):
        """Train model with k-fold cross validation and save best model"""
        
        kf = KFold(n_splits=fold_size, random_state=42, shuffle=True)
        MRR = []
        MRR_AT = 100
        max_MRR = 0
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        for epoch, (train_index, dev_index) in enumerate(kf.split(pos_pairs)):
            pos_train_samples = [pos_pairs[i] for i in train_index]
            pos_dev_samples = [pos_pairs[i] for i in dev_index]
            
            train_dataloader = self.prepare_training_data(
                pos_train_samples, 
                neg_pairs,
                batch_size
            )
            
            # Calculate warmup steps as in paper (10% of training data)
            warmup_steps = int(len(pos_train_samples) * 0.1)
            
            # Use regression objective (cosine similarity) as recommended in paper
            train_loss = losses.CosineSimilarityLoss(self.sbert_model)
            
            # Train for this fold
            self.sbert_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': lr},
                output_path=output_path,
                show_progress_bar=True
            )
            
            # Evaluate using MRR
            try:
                testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
                encodings_short = self.sbert_model.encode(
                    [text[:512] for text in testads["short_texts"]], 
                    show_progress_bar=True
                )
                testads["embeddings_short"] = encodings_short.tolist()
                
                embeddings = encode_jobs(self.sbert_model)
                
                # Calculate similarities and MRR
                for k in embeddings.keys():
                    similarities = util.cos_sim(testads["embeddings_short"], embeddings[k]["embeddings"])
                    
                    ranks = []
                    missing = 0
                    simdf = pd.DataFrame(similarities, columns=embeddings[k]["esco_id"], index=testads["esco_id"])
                    
                    for i in tqdm(range(len(simdf))):
                        id = simdf.iloc[i].name
                        series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                        rank = (series[series["index"]==id].index.item()+1)
                        
                        if rank > MRR_AT:
                            missing += 1
                            ranks.append(0)
                        else:
                            ranks.append(1/rank)
                            
                    missing = missing/len(simdf)
                    current_mrr = np.mean(ranks)
                    
                    current_run = {
                        "model": output_path.split("/")[-1],
                        "epoch": epoch,
                        "embedding_kind": k,
                        "MRR": current_mrr,
                        "missing": missing,
                        "MRR@": MRR_AT,
                        "training_details": [training_start, batch_size, lr, warmup_steps, num_epochs, fold_size]
                    }
                    MRR.append(current_run)
                    
                    if current_mrr > max_MRR:
                        print(f"New best Model saved after epoch {epoch}")
                        max_MRR = current_mrr
                        self.sbert_model.save(f"{output_path}_best")
                        with open(f"{output_path}/model_info.json", 'w') as f:
                            json.dump(current_run, f)
                    
                    pd.DataFrame(MRR).to_excel(f"{output_path}/eval/{training_start}_training_details.xlsx")
                    
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                continue
        
        self.sbert_model.save(output_path)
        return MRR

    def encode(self, texts: List[str]):
        """Encode texts to embeddings"""
        return self.sbert_model.encode([text[:512] for text in texts], convert_to_tensor=True)

# Example usage
def train_consultantbert(pos_pairs, neg_pairs, output_path):
    """Main training function with paper's exact parameters"""
    model = ConsultantBERT()
    
    mrr_results = model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path=output_path,
        batch_size=4,  # As per paper
        num_epochs=5,  # As per paper
        fold_size=10,
        lr=2e-5
    )
    
    return model, mrr_results

def create_training_subset(pos_pairs, neg_pairs, sample_size=1000):
    """
    Create a smaller training subset for testing
    
    Args:
        pos_pairs: List of positive pairs
        neg_pairs: List of negative pairs
        sample_size: Number of pairs to sample from each
    """
    # Set seed for reproducibility
    random.seed(42)
    
    # Sample from positive and negative pairs
    pos_sample = random.sample(pos_pairs, min(sample_size, len(pos_pairs)))
    neg_sample = random.sample(neg_pairs, min(sample_size, len(neg_pairs)))
    
    print(f"Created training subset:")
    print(f"Original positive pairs: {len(pos_pairs)} -> Sampled: {len(pos_sample)}")
    print(f"Original negative pairs: {len(neg_pairs)} -> Sampled: {len(neg_sample)}")
    
    return pos_sample, neg_sample

if __name__ == "__main__":
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])

    # Create smaller subset for testing
    pos_sample, neg_sample = create_training_subset(pos_pairs, neg_pairs, sample_size=1000)
    
    # Train on smaller subset
    model, mrr_results = train_consultantbert(
        pos_pairs=pos_sample,
        neg_pairs=neg_sample,
        output_path="../00_data/SBERT_Models/models/consultantbert_multilingual_test"
    )