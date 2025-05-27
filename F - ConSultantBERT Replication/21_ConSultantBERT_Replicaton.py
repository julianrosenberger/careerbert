"""
ConsultantBERT Implementation - Faithful to Lavi et al. (2021) Paper
Supports both paper-exact replication and extended modes with MNR loss
"""

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
    ConsultantBERT implementation faithful to Lavi et al. (2021)
    "conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers"
    
    Supports:
    - Paper-exact replication (paper_mode=True)
    - Extended modes with MNR loss (paper_mode=False)
    """
    
    def __init__(self, 
                 mode: str = "regressor",
                 paper_mode: bool = True,
                 model_name: str = "google-bert/bert-base-multilingual-cased", 
                 max_seq_length: int = 512):
        """
        Initialize ConsultantBERT model
        
        Args:
            mode: "classifier", "regressor", or "mnr" (MNR only available when paper_mode=False)
            paper_mode: If True, uses exact paper implementation. If False, allows extensions
            model_name: Base BERT model (paper uses bert-base-multilingual-cased)
            max_seq_length: Maximum sequence length (paper uses 512)
        """
        set_seed(42)
        self.mode = mode
        self.paper_mode = paper_mode
        self.config = Config() if 'Config' in globals() else None
        
        # Validate mode
        if paper_mode and mode not in ["classifier", "regressor"]:
            raise ValueError("Paper mode only supports 'classifier' and 'regressor' modes")
        elif not paper_mode and mode not in ["classifier", "regressor", "mnr", "rf_hybrid"]:
            raise ValueError("Extended mode supports 'classifier', 'regressor', 'mnr', and 'rf_hybrid'")
        
        # Create bi-encoder architecture as described in paper
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        self.embedding_dim = word_embedding_model.get_word_embedding_dimension()
        
        # Paper uses mean pooling specifically
        pooling_model = models.Pooling(
            self.embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.rf_model = None  # For RF hybrid mode (extension)
        
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
    
    def prepare_training_data_paper(self, pos_pairs, neg_pairs, batch_size=4):
        """Prepare training data exactly as described in the paper"""
        train_examples = []
        
        if self.mode == "classifier":
            # Paper: "classification objective ... cross-entropy loss"
            # Create balanced dataset with positive and negative pairs
            neg_pairs_dict = {}
            for anchor, negative in neg_pairs:
                if anchor not in neg_pairs_dict:
                    neg_pairs_dict[anchor] = []
                neg_pairs_dict[anchor].append(negative)
            
            print("Creating training examples for classifier mode (paper-exact)...")
            for anchor, positive in tqdm(pos_pairs):
                # Add positive example (label 1)
                train_examples.append(InputExample(
                    texts=[anchor, positive],
                    label=1
                ))
                
                # Add one negative example per positive (label 0)
                if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                    train_examples.append(InputExample(
                        texts=[anchor, neg_pairs_dict[anchor][0]],
                        label=0
                    ))
        
        elif self.mode == "regressor":
            # Paper: "regression loss, which is MSE loss between the cosine similarity score and true similarity label"
            print("Creating training examples for regressor mode (paper-exact)...")
            
            # Create negative pairs dictionary for efficient lookup
            neg_pairs_dict = {}
            for anchor, negative in neg_pairs:
                if anchor not in neg_pairs_dict:
                    neg_pairs_dict[anchor] = []
                neg_pairs_dict[anchor].append(negative)
            
            # Add positive pairs with similarity label 1.0
            for anchor, positive in tqdm(pos_pairs):
                train_examples.append(InputExample(
                    texts=[anchor, positive],
                    label=1.0  # True similarity for positive pairs
                ))
                
                # Add corresponding negative pair with similarity label 0.0
                if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                    train_examples.append(InputExample(
                        texts=[anchor, neg_pairs_dict[anchor][0]],
                        label=0.0  # No similarity for negative pairs
                    ))
        
        print(f"Created {len(train_examples)} training examples")
        return DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    def prepare_training_data_extended(self, pos_pairs, neg_pairs, batch_size=4):
        """Extended training data preparation for non-paper modes"""
        train_examples = []
        
        if self.mode in ["mnr", "rf_hybrid"]:
            # For MNR modes, use simple pairs without labels
            print(f"Creating training examples for {self.mode} mode...")
            for item in pos_pairs:
                train_examples.append(InputExample(texts=[item[0], item[1]]))
        
        print(f"Created {len(train_examples)} training examples")
        return DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    def get_loss_function_paper(self):
        """Get loss function exactly as described in the paper"""
        if self.mode == "classifier":
            # Paper: "cross-entropy loss"
            return losses.SoftmaxLoss(
                model=self.sbert_model,
                sentence_embedding_dimension=self.embedding_dim,
                num_labels=2
            )
        elif self.mode == "regressor":
            # Paper: "MSE loss between the cosine similarity score and true similarity label"
            return losses.CosineSimilarityLoss(self.sbert_model)
    
    def get_loss_function_extended(self):
        """Get loss function for extended modes"""
        if self.mode == "classifier":
            return losses.SoftmaxLoss(
                model=self.sbert_model,
                sentence_embedding_dimension=self.embedding_dim,
                num_labels=2
            )
        elif self.mode == "regressor":
            return losses.CosineSimilarityLoss(self.sbert_model)
        elif self.mode in ["mnr", "rf_hybrid"]:
            return losses.MultipleNegativesRankingLoss(self.sbert_model)
    
    def evaluate_paper_metrics(self, test_pairs):
        """
        Evaluate using paper's exact metrics: ROC-AUC, precision, recall, F1
        
        Args:
            test_pairs: List of tuples (anchor, positive/negative, label)
        """
        y_true = []
        y_pred = []
        
        print("Evaluating using paper metrics (ROC-AUC, precision, recall, F1)...")
        for anchor, text, label in tqdm(test_pairs):
            # Get embeddings and compute cosine similarity
            anchor_emb = self.sbert_model.encode([anchor])
            text_emb = self.sbert_model.encode([text])
            similarity = util.cos_sim(anchor_emb, text_emb).item()
            
            y_true.append(label)
            y_pred.append(similarity)
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred)
        
        # For precision/recall, need to convert similarities to binary predictions
        # Use median as threshold (or optimize this)
        threshold = np.median(y_pred)
        y_pred_binary = [1 if sim > threshold else 0 for sim in y_pred]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='macro', zero_division=0
        )
        
        return {
            "ROC-AUC": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold_used": threshold
        }
    
    def create_test_pairs_from_splits(self, pos_test, neg_test):
        """Convert test splits into format needed for paper evaluation"""
        test_pairs = []
        
        # Add positive test pairs
        for anchor, positive in pos_test:
            test_pairs.append((anchor, positive, 1))
        
        # Add negative test pairs  
        for anchor, negative in neg_test:
            test_pairs.append((anchor, negative, 0))
        
        return test_pairs
    
    def evaluate_esco_metrics(self, output_path, training_start):
        """Extended evaluation using ESCO embeddings (not in paper)"""
        if self.paper_mode:
            print("Skipping ESCO evaluation in paper mode (not in original paper)")
            return []
        
        print("Evaluating with ESCO embeddings (extended mode)...")
        
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
    
    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,        # Paper exact
                   num_epochs: int = 5,        # Paper exact  
                   lr: float = 2e-5,           # Paper exact
                   test_size: float = 0.2):    # Paper uses 80/20 split
        """
        Main training function that handles both paper-exact and extended modes
        
        Paper training parameters:
        - batch_size=4, num_epochs=5, lr=2e-5, 80/20 train/dev split
        """
        print(f"Training ConsultantBERT in {self.mode} mode")
        print(f"Paper mode: {self.paper_mode}")
        print(f"Positive pairs: {len(pos_pairs)}, Negative pairs: {len(neg_pairs)}")
        
        # Initialize tracking
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        best_metric = 0
        best_model = None
        
        # Create directories
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        # Create train/dev split exactly as paper (80/20)
        pos_train_samples, pos_dev_samples = train_test_split(
            pos_pairs, test_size=test_size, random_state=42
        )
        
        # For paper evaluation, also need test split from negative pairs
        if self.paper_mode:
            neg_train_samples, neg_dev_samples = train_test_split(
                neg_pairs, test_size=test_size, random_state=42
            )
            print(f"Training: {len(pos_train_samples)} pos, {len(neg_train_samples)} neg")
            print(f"Dev: {len(pos_dev_samples)} pos, {len(neg_dev_samples)} neg")
        else:
            print(f"Training samples: {len(pos_train_samples)}, Dev samples: {len(pos_dev_samples)}")
        
        # Prepare training data
        if self.paper_mode:
            train_dataloader = self.prepare_training_data_paper(pos_train_samples, neg_pairs, batch_size)
            loss_function = self.get_loss_function_paper()
        else:
            train_dataloader = self.prepare_training_data_extended(pos_train_samples, neg_pairs, batch_size)
            loss_function = self.get_loss_function_extended()
        
        # Calculate warmup steps (paper: 10% of training data)
        warmup_steps = int(len(pos_train_samples) * 0.1)
        
        # Set up evaluator for development set (used during training)
        if not self.paper_mode:
            dev_set = self.create_development_set(pos_dev_samples, neg_pairs)
            evaluator = evaluation.RerankingEvaluator(dev_set, at_k=100, show_progress_bar=True)
        else:
            evaluator = None  # Paper doesn't specify using evaluator during training
        
        # Train the model with paper-exact parameters
        print("Starting SBERT training...")
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, loss_function)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            checkpoint_path=f"{output_path}/modeltrain" if not self.paper_mode else None,
            checkpoint_save_total_limit=1,
            optimizer_params={'lr': lr},
            checkpoint_save_steps=1000,
            output_path=output_path
        )
        
        # Evaluate model
        if self.paper_mode:
            # Paper evaluation: ROC-AUC, precision, recall, F1
            test_pairs = self.create_test_pairs_from_splits(pos_dev_samples, neg_dev_samples)
            paper_results = self.evaluate_paper_metrics(test_pairs)
            
            print("\nPaper Evaluation Results:")
            for metric, value in paper_results.items():
                print(f"{metric}: {value:.4f}")
            
            # Save paper results
            paper_results_with_metadata = {
                **paper_results,
                "model": f"{output_path.split('/')[-1]}_{self.mode}",
                "mode": self.mode,
                "paper_mode": self.paper_mode,
                "training_params": {
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "warmup_steps": warmup_steps
                }
            }
            
            results_to_return = [paper_results_with_metadata]
            best_metric = paper_results["ROC-AUC"]
            
        else:
            # Extended evaluation: ESCO metrics
            esco_results = self.evaluate_esco_metrics(output_path, training_start)
            
            if esco_results:
                best_metric = max(esco_results, key=lambda x: x['MRR'])['MRR']
                results_to_return = esco_results
            else:
                results_to_return = []
        
        # Save best model
        if best_metric > 0:
            best_model = self.sbert_model
            best_model.save(f"{output_path}_best")
            
            # Save model info
            model_info = {
                "best_metric": best_metric,
                "mode": self.mode,
                "paper_mode": self.paper_mode,
                "training_params": {
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "warmup_steps": warmup_steps
                }
            }
            write_json(f"{output_path}/model_info.json", model_info)
        
        # Save results
        if results_to_return:
            results_df = pd.DataFrame(results_to_return)
            mode_suffix = "paper" if self.paper_mode else "extended"
            results_df.to_excel(f"{output_path}/eval/{training_start}_{self.mode}_{mode_suffix}_results.xlsx")
        
        # Save final model
        self.sbert_model.save(output_path)
        
        print(f"\nTraining completed! Best metric: {best_metric:.4f}")
        print(f"Model saved to: {output_path}")
        
        return results_to_return, best_model

def create_training_subset(pos_pairs, neg_pairs, sample_size=1000):
    """Create a smaller training subset for testing"""
    random.seed(42)
    pos_sample = random.sample(pos_pairs, min(sample_size, len(pos_pairs)))
    neg_sample = random.sample(neg_pairs, min(sample_size, len(neg_pairs)))
    
    print(f"Created training subset:")
    print(f"Positive pairs: {len(pos_pairs)} -> {len(pos_sample)}")
    print(f"Negative pairs: {len(neg_pairs)} -> {len(neg_sample)}")
    
    return pos_sample, neg_sample

def replicate_paper_results(pos_pairs, neg_pairs, base_output_path, use_subset=True):
    """
    Replicate the exact results from Lavi et al. (2021) paper
    Tests both classifier and regressor modes as described in the paper
    """
    print("="*80)
    print("REPLICATING CONSULTANTBERT PAPER RESULTS")
    print("Lavi et al. (2021): conSultantBERT: Fine-tuned Siamese Sentence-BERT")
    print("="*80)
    
    if use_subset:
        pos_pairs, neg_pairs = create_training_subset(pos_pairs, neg_pairs, 1000)
    
    paper_modes = ["classifier", "regressor"]
    results = {}
    
    for mode in paper_modes:
        print(f"\n{'='*60}")
        print(f"PAPER REPLICATION: {mode.upper()} MODE")
        print(f"{'='*60}")
        
        model = ConsultantBERT(mode=mode, paper_mode=True)
        output_path = f"{base_output_path}_paper_{mode}"
        
        # Use exact paper parameters
        results_list, best_model = model.train_model(
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            output_path=output_path,
            batch_size=4,      # Paper exact
            num_epochs=5,      # Paper exact  
            lr=2e-5,          # Paper exact
            test_size=0.2     # Paper exact (80/20 split)
        )
        
        if results_list:
            results[f"paper_{mode}"] = results_list[0]
            print(f"\nPaper {mode} results:")
            for metric, value in results_list[0].items():
                if isinstance(value, (int, float)) and metric != "training_params":
                    print(f"  {metric}: {value:.4f}")
    
    # Print comparison as in paper Table 1
    print(f"\n{'='*80}")
    print("PAPER REPLICATION RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<40} {'ROC-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    
    for mode_key, result in results.items():
        model_name = f"conSultantBERT{mode_key.split('_')[1].title()}"
        print(f"{model_name:<40} {result['ROC-AUC']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f}")
    
    return results

def compare_paper_vs_extended(pos_pairs, neg_pairs, base_output_path, use_subset=True):
    """
    Compare paper implementation vs extended implementation with MNR loss
    """
    print("="*80)
    print("COMPARING PAPER VS EXTENDED IMPLEMENTATIONS")
    print("="*80)
    
    if use_subset:
        pos_pairs, neg_pairs = create_training_subset(pos_pairs, neg_pairs, 1000)
    
    modes_to_test = [
        ("regressor", True, "Paper Regressor"),    # Paper best
        ("regressor", False, "Extended Regressor"), # Extended regressor  
        ("mnr", False, "Extended MNR"),            # Extended MNR
    ]
    
    results = {}
    
    for mode, paper_mode, description in modes_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING: {description}")
        print(f"{'='*60}")
        
        model = ConsultantBERT(mode=mode, paper_mode=paper_mode)
        output_path = f"{base_output_path}_{'paper' if paper_mode else 'extended'}_{mode}"
        
        results_list, best_model = model.train_model(
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            output_path=output_path,
            batch_size=4,
            num_epochs=5,
            lr=2e-5
        )
        
        if results_list:
            results[description] = results_list[0] if paper_mode else results_list
            
    return results

def main():
    """Main function to run ConsultantBERT experiments"""
    # Load data
    print("Loading training data...")
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    print(f"Loaded {len(pos_pairs)} positive and {len(neg_pairs)} negative pairs")
    
    base_output_path = "../00_data/SBERT_Models/models/consultantbert_multilingual"
    
    # Option 1: Replicate paper exactly
    paper_results = replicate_paper_results(pos_pairs, neg_pairs, base_output_path, use_subset=True)
    
    # Option 2: Compare paper vs extended
    # comparison_results = compare_paper_vs_extended(pos_pairs, neg_pairs, base_output_path, use_subset=True)
    
    return paper_results

if __name__ == "__main__":
    results = main()