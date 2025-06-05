from sentence_transformers import SentenceTransformer, InputExample, models, losses, util, evaluation
from torch.utils.data import DataLoader
import torch
from transformers import set_seed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import json
import random
import pickle
from helpers import *

class ConsultantBERT:
    """
    ConsultantBERT implementation as used in CareerBERT paper
    
    Modes:
    - "classifier": Paper's classification objective (SoftmaxLoss)
    - "regressor": Paper's regression objective (CosineSimilarityLoss) 
    - "mnr": Multiple Negatives Ranking Loss (our addition)
    - "rf_hybrid": Random Forest on top of embeddings (our addition)
    """
    
    def __init__(self, 
                 mode: str = "regressor",
                 model_name: str = "google-bert/bert-base-multilingual-cased", 
                 max_seq_length: int = 512):
        set_seed(42)
        self.mode = mode
        self.config = Config() if 'Config' in globals() else None
        
        if mode not in ["classifier", "regressor", "mnr", "rf_hybrid"]:
            raise ValueError("Supported modes: classifier, regressor, mnr, rf_hybrid")
        
        # Create bi-encoder architecture
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        self.embedding_dim = word_embedding_model.get_word_embedding_dimension()
        
        pooling_model = models.Pooling(
            self.embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.rf_model = None
        
    def create_development_set(self, pos_dev_samples, neg_pairs):
        """Create development set with query/positive/negative structure for evaluation"""
        dev_set_total = []
        anchors = set([x[0] for x in pos_dev_samples])
        neg_dev_samples = [x for x in neg_pairs if x[0] in anchors]
        
        print("Creating development set...")
        for anchor in tqdm(anchors):
            pos_pairs_filtered = [x[1] for x in pos_dev_samples if x[0] == anchor]
            neg_pairs_filtered = [x[1] for x in neg_dev_samples if x[0] == anchor]
            
            if pos_pairs_filtered and neg_pairs_filtered:
                dev_set_total.append({
                    "query": anchor, 
                    "positive": pos_pairs_filtered, 
                    "negative": neg_pairs_filtered
                })
        
        print(f"Created development set with {len(dev_set_total)} query groups")
        return dev_set_total
    
    def prepare_training_data(self, pos_pairs, neg_pairs, batch_size=4):
        """Prepare training data based on mode"""
        train_examples = []
        
        if self.mode == "classifier":
            # Paper: binary classification with cross-entropy
            neg_pairs_dict = {}
            for anchor, negative in neg_pairs:
                if anchor not in neg_pairs_dict:
                    neg_pairs_dict[anchor] = []
                neg_pairs_dict[anchor].append(negative)
            
            print("Creating training examples for classifier mode...")
            for anchor, positive in tqdm(pos_pairs):
                train_examples.append(InputExample(texts=[anchor, positive], label=1))
                
                if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                    train_examples.append(InputExample(
                        texts=[anchor, neg_pairs_dict[anchor][0]], label=0
                    ))
        
        elif self.mode == "regressor":
            # Paper: regression with MSE on cosine similarity
            neg_pairs_dict = {}
            for anchor, negative in neg_pairs:
                if anchor not in neg_pairs_dict:
                    neg_pairs_dict[anchor] = []
                neg_pairs_dict[anchor].append(negative)
            
            print("Creating training examples for regressor mode...")
            for anchor, positive in tqdm(pos_pairs):
                train_examples.append(InputExample(texts=[anchor, positive], label=1.0))
                
                if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                    train_examples.append(InputExample(
                        texts=[anchor, neg_pairs_dict[anchor][0]], label=0.0
                    ))
        
        elif self.mode in ["mnr", "rf_hybrid"]:
            # Our extension: MNR loss
            print(f"Creating training examples for {self.mode} mode...")
            for item in pos_pairs:
                train_examples.append(InputExample(texts=[item[0], item[1]]))
        
        print(f"Created {len(train_examples)} training examples")
        return DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    def get_loss_function(self):
        """Get appropriate loss function for the mode"""
        if self.mode == "classifier":
            return losses.SoftmaxLoss(
                model=self.sbert_model,
                sentence_embedding_dimension=self.embedding_dim,
                num_labels=2
            )
        elif self.mode == "regressor":
            # FIXED: Use CosineSimilarityLoss for paper's regression objective
            return losses.CosineSimilarityLoss(self.sbert_model)
        elif self.mode in ["mnr", "rf_hybrid"]:
            return losses.MultipleNegativesRankingLoss(self.sbert_model)
    
    def evaluate_esco_metrics(self, output_path, training_start):
        """Evaluate using ESCO embeddings for job matching"""
        print("Evaluating with ESCO embeddings...")
        
        try:
            # Load test data
            testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
            encodings_short = self.sbert_model.encode(list(testads["short_texts"]), show_progress_bar=True)
            testads["embeddings_short"] = encodings_short.tolist()
            
            # Generate ESCO embeddings
            embeddings = encode_jobs(self.sbert_model)
            
            MRR = []
            MRR_AT = 100
            
            # Evaluate each embedding type
            for k in embeddings:
                print(f"Evaluating {k} embeddings...")
                similarities = util.cos_sim(testads["embeddings_short"], embeddings[k]["embeddings"])
                
                ranks = []
                missing = 0
                simdf = pd.DataFrame(similarities, columns=embeddings[k]["esco_id"], index=testads["esco_id"])
                
                for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
                    id = simdf.iloc[i].name
                    series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                    rank = (series[series["index"] == id].index.item() + 1)
                    
                    if rank > MRR_AT:
                        missing += 1
                        ranks.append(0)
                    else:
                        ranks.append(1/rank)
                
                missing = missing / len(simdf)
                
                result = {
                    "model": f"{output_path.split('/')[-1]}_{self.mode}",
                    "embedding_kind": k,
                    "MRR": np.mean(ranks),
                    "missing": missing,
                    "MRR@": MRR_AT,
                    "mode": self.mode,
                    "training_timestamp": training_start
                }
                MRR.append(result)
                
                print(f"MRR for {k}: {np.mean(ranks):.4f}")
            
            return MRR
            
        except Exception as e:
            print(f"ESCO evaluation failed: {e}")
            print("Continuing without ESCO metrics...")
            return []
    
    def train_rf_hybrid(self, pos_pairs, neg_pairs, test_size=0.2):
        """Train Random Forest classifier on SBERT embeddings"""
        print("Training Random Forest on SBERT embeddings...")
        
        # Create training data
        X = []
        y = []
        
        print("Encoding positive pairs...")
        for anchor, positive in tqdm(pos_pairs):
            anchor_emb = self.sbert_model.encode([anchor])
            positive_emb = self.sbert_model.encode([positive])
            combined_emb = np.concatenate([anchor_emb[0], positive_emb[0]])
            X.append(combined_emb)
            y.append(1)
        
        print("Encoding negative pairs...")
        for anchor, negative in tqdm(neg_pairs[:len(pos_pairs)]):  # Balance dataset
            anchor_emb = self.sbert_model.encode([anchor])
            negative_emb = self.sbert_model.encode([negative])
            combined_emb = np.concatenate([anchor_emb[0], negative_emb[0]])
            X.append(combined_emb)
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred)
        
        print(f"RF Hybrid ROC-AUC: {roc_auc:.4f}")
        return {"RF_ROC_AUC": roc_auc}
    
    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,
                   num_epochs: int = 5,
                   lr: float = 2e-5,
                   test_size: float = 0.2):
        """Main training function"""
        print(f"Training ConsultantBERT in {self.mode} mode")
        print(f"Positive pairs: {len(pos_pairs)}, Negative pairs: {len(neg_pairs)}")
        
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        best_metric = 0
        best_model = None
        
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        # Create train/dev split
        pos_train_samples, pos_dev_samples = train_test_split(
            pos_pairs, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(pos_train_samples)}, Dev samples: {len(pos_dev_samples)}")
        
        if self.mode == "rf_hybrid":
            # Special handling for RF hybrid
            rf_results = self.train_rf_hybrid(pos_pairs, neg_pairs)
            esco_results = self.evaluate_esco_metrics(output_path, training_start)
            
            # Add RF results to ESCO results
            for result in esco_results:
                result.update(rf_results)
            
            return esco_results, self.sbert_model
        
        # Regular SBERT training
        train_dataloader = self.prepare_training_data(pos_train_samples, neg_pairs, batch_size)
        loss_function = self.get_loss_function()
        
        warmup_steps = int(len(pos_train_samples) * 0.1)
        
        # Set up evaluator for development
        dev_set = self.create_development_set(pos_dev_samples, neg_pairs)
        evaluator = evaluation.RerankingEvaluator(dev_set, at_k=100, show_progress_bar=True)
        
        print("Starting SBERT training...")
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, loss_function)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            checkpoint_path=f"{output_path}/modeltrain",
            checkpoint_save_total_limit=1,
            optimizer_params={'lr': lr},
            checkpoint_save_steps=1000,
            output_path=output_path
        )
        
        # Evaluate with ESCO metrics
        esco_results = self.evaluate_esco_metrics(output_path, training_start)
        
        if esco_results:
            best_metric = max(esco_results, key=lambda x: x['MRR'])['MRR']
            best_model = self.sbert_model
            best_model.save(f"{output_path}_best")
            
            # Save model info
            model_info = {
                "best_metric": best_metric,
                "mode": self.mode,
                "training_params": {
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "warmup_steps": warmup_steps
                }
            }
            write_json(f"{output_path}/model_info.json", model_info)
        
        # Save results
        if esco_results:
            results_df = pd.DataFrame(esco_results)
            results_df.to_excel(f"{output_path}/eval/{training_start}_{self.mode}_results.xlsx")
        
        # Save final model
        self.sbert_model.save(output_path)
        
        print(f"\nTraining completed! Best MRR: {best_metric:.4f}")
        print(f"Model saved to: {output_path}")
        
        return esco_results, best_model

def create_training_subset(pos_pairs, neg_pairs, sample_size=1000):
    """Create a smaller training subset for testing"""
    random.seed(42)
    pos_sample = random.sample(pos_pairs, min(sample_size, len(pos_pairs)))
    neg_sample = random.sample(neg_pairs, min(sample_size, len(neg_pairs)))
    
    print(f"Created training subset:")
    print(f"Positive pairs: {len(pos_pairs)} -> {len(pos_sample)}")
    print(f"Negative pairs: {len(neg_pairs)} -> {len(neg_sample)}")
    
    return pos_sample, neg_sample

def train_all_consultant_bert_modes(pos_pairs, neg_pairs, base_output_path, use_subset=True):
    """
    Train all ConsultantBERT modes as used in CareerBERT paper
    """
    print("="*80)
    print("TRAINING ALL CONSULTANTBERT MODES (CareerBERT Paper Setup)")
    print("="*80)
    
    if use_subset:
        pos_pairs, neg_pairs = create_training_subset(pos_pairs, neg_pairs, 1000)
    
    modes = ["classifier", "regressor", "mnr", "rf_hybrid"]
    all_results = []
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"TRAINING: ConsultantBERT {mode.upper()} MODE")
        print(f"{'='*60}")
        
        model = ConsultantBERT(mode=mode)
        output_path = f"{base_output_path}_{mode}"
        
        results_list, best_model = model.train_model(
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            output_path=output_path,
            batch_size=4,
            num_epochs=5,
            lr=2e-5
        )
        
        if results_list:
            all_results.extend(results_list)
            
            # Print summary for this mode
            if isinstance(results_list, list) and len(results_list) > 0:
                best_mrr = max(results_list, key=lambda x: x['MRR'])['MRR']
                print(f"Best MRR for {mode}: {best_mrr:.4f}")
    
    # Create comparison summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Group by mode and get best MRR for each
        summary = results_df.groupby('mode')['MRR'].max().reset_index()
        summary = summary.sort_values('MRR', ascending=False)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Mode':<15} {'Best MRR':<10}")
        print("-"*25)
        for _, row in summary.iterrows():
            print(f"{row['mode']:<15} {row['MRR']:<10.4f}")
        
        # Save combined results
        timestamp = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        results_df.to_excel(f"{base_output_path}_all_modes_{timestamp}.xlsx")
        
        return all_results
    
    return []

def main():
    """Main function for ConsultantBERT training as used in CareerBERT paper"""
    
    # Load data
    print("Loading training data...")
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    print(f"Loaded {len(pos_pairs)} positive and {len(neg_pairs)} negative pairs")
    
    base_output_path = "../00_data/SBERT_Models/models/consultantbert_multilingual"
    
    # Train all modes as used in CareerBERT paper
    results = train_all_consultant_bert_modes(
        pos_pairs, neg_pairs, base_output_path, use_subset=True
    )
    
    return results

if __name__ == "__main__":
    results = main()