from sentence_transformers import SentenceTransformer, InputExample, models, losses, util, evaluation
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModel, set_seed
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
import json
import random
from helpers import *

def create_training_samples(pos_dev_samples, neg_pairs):
    """Create development set with query/positive/negative structure"""
    dev_set_total = []
    anchors = set([x[0] for x in pos_dev_samples])
    neg_dev_samples = [x for x in neg_pairs if x[0] in anchors]
    print("Creating Devset")
    for anchor in tqdm(anchors):
        pos_pairs_filtered = [x[1] for x in pos_dev_samples if x[0]==anchor]
        neg_pairs_filtered = [x[1] for x in neg_dev_samples if x[0]==anchor]
        dev_set_total.append({"query": anchor, "positive": pos_pairs_filtered, "negative": neg_pairs_filtered})
    return dev_set_total

class ConsultantBERT:
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased", max_seq_length: int = 512):
        set_seed(42)
        
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True  # As per consultantBERT paper
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,  # As per consultantBERT paper
                   num_epochs: int = 5,  # As per consultantBERT paper
                   lr: float = 2e-5):
        """Train model using 80:20 split as in consultantBERT paper but with jobGBERT evaluation"""
        
        # Initialize tracking variables
        MRR = []
        MRR_AT = 100
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        max_MRR = 0
        
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        # Create 80:20 split as per consultantBERT paper
        pos_train_samples, pos_dev_samples = train_test_split(pos_pairs, test_size=0.2, random_state=42)
        print(f"Training samples: {len(pos_train_samples)}, Development samples: {len(pos_dev_samples)}")
        
        # Create development set
        dev_set_total = create_training_samples(pos_dev_samples, neg_pairs)
        
        # Prepare training data
        train_examples = []
        for item in pos_train_samples:
            train_examples.append(InputExample(texts=[item[0], item[1]]))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps (10% of training data)
        warmup = len(pos_train_samples) * 0.1
        
        # Initialize loss and evaluator as in jobGBERT
        train_loss = losses.MultipleNegativesRankingLoss(self.sbert_model)
        evaluator = evaluation.RerankingEvaluator(dev_set_total, at_k=100, show_progress_bar=True)
        
        # Train the model
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup,
            evaluator=evaluator,
            checkpoint_path=f"{output_path}/modeltrain",
            checkpoint_save_total_limit=1,
            optimizer_params={'lr': lr},
            checkpoint_save_steps=1000,
            output_path=output_path
        )
        
        # Evaluate using ESCO embeddings (jobGBERT approach)
        testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
        encodings_short = self.sbert_model.encode(list(testads["short_texts"]), show_progress_bar=True)
        testads["embeddings_short"] = encodings_short.tolist()
        embeddings = encode_jobs(self.sbert_model)
        
        # Calculate similarities and MRR
        similarities = {}
        for k in embeddings:
            similarities[k] = util.cos_sim(testads["embeddings_short"], embeddings[k]["embeddings"])
            
        for k in similarities.keys():
            ranks = []
            missing = 0
            simdf = pd.DataFrame(similarities[k], columns=embeddings[k]["esco_id"], index=testads["esco_id"])
            
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
            current_run = {
                "model": output_path.split("/")[-1],
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "MRR@": MRR_AT,
                "training_details": [training_start, batch_size, lr, warmup, num_epochs]
            }
            MRR.append(current_run)
            
            df = pd.DataFrame(MRR)
            print("\nCurrent Results:")
            print(df)
            
            # Save if best model
            if np.mean(ranks) > max_MRR:
                print(f"New best Model saved")
                max_MRR = np.mean(ranks)
                self.sbert_model.save(f"{output_path}_best")
                write_json(f"{output_path}/model_info.json", current_run)
            
            df.to_excel(f"{output_path}/eval/{training_start}_training_details.xlsx")
        
        # Save final model
        self.sbert_model.save(output_path)
        return MRR

if __name__ == "__main__":
    # Load data
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    # Optional: Use smaller subset for testing
    sample_size = 1000  # Comment out these lines for full training
    random.seed(42)
    pos_pairs = random.sample(pos_pairs, min(sample_size, len(pos_pairs)))
    neg_pairs = random.sample(neg_pairs, min(sample_size, len(neg_pairs)))
    
    print(f"Training with {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
    
    # Train model
    model = ConsultantBERT()
    mrr_results = model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/SBERT_Models/models/consultantbert_multilingual",
        batch_size=4,  # As per consultantBERT paper
        num_epochs=5,  # As per consultantBERT paper
        lr=2e-5
    )