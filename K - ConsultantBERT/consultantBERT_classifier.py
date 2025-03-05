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
        self.embedding_dim = word_embedding_model.get_word_embedding_dimension()
        pooling_model = models.Pooling(
            self.embedding_dim,
            pooling_mode_mean_tokens=True  # As per consultantBERT paper
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,  # As per consultantBERT paper
                   num_epochs: int = 5,  # As per consultantBERT paper
                   lr: float = 2e-5):
        """Train model using softmax classifier but evaluate with ESCO/EURES"""
        
        # Initialize tracking variables
        MRR = []
        MRR_AT = 100
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        max_MRR = 0
        best_model_to_save = None
        
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        # Create 80:20 split
        pos_train_samples, pos_dev_samples = train_test_split(pos_pairs, test_size=0.2, random_state=42)
        print(f"Training samples: {len(pos_train_samples)}, Development samples: {len(pos_dev_samples)}")
        
        # Create development set for ESCO evaluation
        dev_set_total = create_training_samples(pos_dev_samples, neg_pairs)
        
        # Create negative pairs dictionary for faster lookup
        print("Creating negative pairs dictionary...")
        neg_pairs_dict = {}
        for anchor, negative in neg_pairs:
            if anchor not in neg_pairs_dict:
                neg_pairs_dict[anchor] = []
            neg_pairs_dict[anchor].append(negative)
        
        # Prepare training data with balanced positive and negative pairs
        print("Creating training examples...")
        train_examples = []
        for anchor, positive in tqdm(pos_train_samples):
            # Add positive example
            train_examples.append(InputExample(
                texts=[anchor, positive],
                label=1
            ))
            
            # Add one negative example per positive
            if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                train_examples.append(InputExample(
                    texts=[anchor, neg_pairs_dict[anchor][0]],
                    label=0
                ))
        
        print(f"Created {len(train_examples)} training examples")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps (10% of training data)
        warmup = len(train_examples) * 0.1
        
        # Initialize softmax loss and evaluator
        train_loss = losses.SoftmaxLoss(
            model=self.sbert_model,
            sentence_embedding_dimension=self.embedding_dim,
            num_labels=2  # Binary classification
        )
        evaluator = evaluation.RerankingEvaluator(dev_set_total, at_k=100, show_progress_bar=True)
        
        # Train the model
        print("Starting training...")
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
        
        # Evaluate using ESCO embeddings
        print("Evaluating with ESCO embeddings...")
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
            
            # Save best model
            if np.mean(ranks) > max_MRR:
                print(f"New best Model saved")
                max_MRR = np.mean(ranks)
                best_model_to_save = self.sbert_model
                best_model_to_save.save(f"{output_path}_best")
                write_json(f"{output_path}/model_info.json", current_run)
            
            df.to_excel(f"{output_path}/eval/{training_start}_training_details.xlsx")
        
        # Save final model
        if best_model_to_save is not None:
            best_model_to_save.save(output_path)
        else:
            self.sbert_model.save(output_path)
        return MRR, best_model_to_save

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
    mrr_results, best_model = model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/SBERT_Models/models/consultantbert_multilingual_softmax",
        batch_size=4,  # As per consultantBERT paper
        num_epochs=5,  # As per consultantBERT paper
        lr=2e-5
    )
    
    # Print final results
    print("\nTraining completed")
    df = pd.DataFrame(mrr_results)
    print("\nFinal Results:")
    print(df)