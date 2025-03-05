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
from sklearn.ensemble import RandomForestClassifier
import os
import json
import random
from helpers import load_json, write_json, encode_jobs, flatten_list, load_data_pairs

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

class ConsultantBERTRegressor:
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased", max_seq_length: int = 512):
        set_seed(42)
        
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True  # As per consultantBERT paper
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.rf_model = None

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,  # As per consultantBERT paper
                   num_epochs: int = 5,  # As per consultantBERT paper
                   lr: float = 2e-5):
        """Train model using regressor approach with both cosine similarity and RF"""
        
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
        
        # Create development set
        dev_set_total = create_training_samples(pos_dev_samples, neg_pairs)
        
        # Prepare training data
        train_examples = []
        for item in pos_train_samples:
            train_examples.append(InputExample(texts=[item[0], item[1]]))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps
        warmup = len(pos_train_samples) * 0.1
        
        # Initialize loss and evaluator
        train_loss = losses.MultipleNegativesRankingLoss(self.sbert_model)
        evaluator = evaluation.RerankingEvaluator(dev_set_total, at_k=100, show_progress_bar=True)
        
        # Train the SBERT model
        print("Training SBERT model...")
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
        
        # Train Random Forest on embeddings
        print("Training Random Forest classifier...")
        X_train = []
        y_train = []
        
        # Prepare data for RF
        print("Preparing data for Random Forest...")
        for anchor, positive in tqdm(pos_train_samples):
            # Get embeddings for positive pair
            anchor_emb = self.sbert_model.encode(anchor)
            positive_emb = self.sbert_model.encode(positive)
            X_train.append(np.concatenate([anchor_emb, positive_emb]))
            y_train.append(1)
            
            # Get embeddings for negative pair if available
            anchor_negatives = [neg[1] for neg in neg_pairs if neg[0] == anchor]
            if anchor_negatives:
                negative_emb = self.sbert_model.encode(anchor_negatives[0])
                X_train.append(np.concatenate([anchor_emb, negative_emb]))
                y_train.append(0)
        
        # Train RF
        print(f"\nTraining Random Forest with {len(X_train)} samples")
        print(f"Positive samples: {sum(y_train)}")
        print(f"Negative samples: {len(y_train) - sum(y_train)}")
        
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Print RF training performance
        train_pred = self.rf_model.predict(X_train)
        train_accuracy = (train_pred == y_train).mean()
        print(f"\nRandom Forest training accuracy: {train_accuracy:.3f}")
        
        # Evaluate using ESCO embeddings
        print("Evaluating with ESCO embeddings...")
        testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
        encodings_short = self.sbert_model.encode(list(testads["short_texts"]), show_progress_bar=True)
        testads["embeddings_short"] = encodings_short.tolist()
        embeddings = encode_jobs(self.sbert_model)
        
        # Calculate similarities and MRR for both approaches
        for approach in ['cosine', 'rf']:
            similarities = {}
            for k in embeddings:
                if approach == 'cosine':
                    print(f"Computing cosine similarities for {k}...")
                    similarities[k] = util.cos_sim(testads["embeddings_short"], embeddings[k]["embeddings"])
                else:  # rf
                    print(f"Computing RF predictions for {k}...")
                    # Prepare data for RF prediction
                    X_test = []
                    n_queries = len(testads["embeddings_short"])
                    n_targets = len(embeddings[k]["embeddings"])
                    
                    print(f"Processing {n_queries} queries x {n_targets} targets = {n_queries * n_targets} pairs")
                    for i, query_emb in enumerate(testads["embeddings_short"]):
                        if i % 100 == 0:
                            print(f"Processing query {i}/{n_queries}")
                        for target_emb in embeddings[k]["embeddings"]:
                            X_test.append(np.concatenate([query_emb, target_emb]))
                    
                    print("Running RF predictions...")
                    rf_pred = self.rf_model.predict_proba(np.array(X_test))[:, 1]  # Get probability of positive class
                    print("Reshaping predictions...")
                    similarities[k] = torch.tensor(rf_pred.reshape(n_queries, n_targets))
                    print(f"RF predictions shape: {similarities[k].shape}")
                
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
                    "model": f"{output_path.split('/')[-1]}_{approach}",
                    "embedding_kind": k,
                    "MRR": np.mean(ranks),
                    "missing": missing,
                    "MRR@": MRR_AT,
                    "training_details": [training_start, batch_size, lr, warmup, num_epochs]
                }
                MRR.append(current_run)
                
                df = pd.DataFrame(MRR)
                print(f"\nCurrent Results ({approach}):")
                print(df)
                
                # Save if best model
                if np.mean(ranks) > max_MRR:
                    print(f"New best Model saved")
                    max_MRR = np.mean(ranks)
                    best_model_to_save = self.sbert_model
                    best_model_to_save.save(f"{output_path}_best")
                    write_json(f"{output_path}/model_info.json", current_run)
                
                df.to_excel(f"{output_path}/eval/{training_start}_training_details.xlsx")
        
        # Save final models
        if best_model_to_save is not None:
            best_model_to_save.save(output_path)
        else:
            self.sbert_model.save(output_path)
            
        # Save RF model
        import pickle
        with open(f"{output_path}/rf_model.pkl", 'wb') as f:
            pickle.dump(self.rf_model, f)
            
        return MRR, best_model_to_save, self.rf_model

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
    model = ConsultantBERTRegressor()
    mrr_results, best_model, rf_model = model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/SBERT_Models/models/consultantbert_multilingual_regressor",
        batch_size=4,
        num_epochs=5,
        lr=2e-5
    )
    
    # Print final results
    print("\nTraining completed")
    df = pd.DataFrame(mrr_results)
    print("\nFinal Results:")
    print(df)

