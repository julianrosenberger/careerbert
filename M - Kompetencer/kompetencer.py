from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
from torch.utils.data import DataLoader
import torch
from transformers import set_seed, RemBertModel, RemBertTokenizer
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
from helpers import *
import sentencepiece as spm


def create_training_samples(pos_dev_samples, neg_pairs):
    """
    Create development set with query/positive/negative structure for evaluation
    
    Args:
        pos_dev_samples: List of positive pairs [(query, positive), ...]
        neg_pairs: List of negative pairs [(query, negative), ...]
    
    Returns:
        List of dicts with format {"query": query, "positive": [pos1, ...], "negative": [neg1, ...]}
    """
    dev_set_total = []
    anchors = set([x[0] for x in pos_dev_samples])
    neg_dev_samples = [x for x in neg_pairs if x[0] in anchors]
    print("Creating development set...")
    for anchor in tqdm(anchors):
        pos_pairs_filtered = [x[1] for x in pos_dev_samples if x[0]==anchor]
        neg_pairs_filtered = [x[1] for x in neg_dev_samples if x[0]==anchor]
        dev_set_total.append({
            "query": anchor, 
            "positive": pos_pairs_filtered, 
            "negative": neg_pairs_filtered
        })
    return dev_set_total

class RemBERTSkillClassifier:
    """
    Implementation of RemBERT for skill classification, following Zhang et al.
    Uses MultipleNegativesRankingLoss and evaluates using Mean Reciprocal Rank (MRR).
    """
    
    def __init__(self, max_seq_length: int = 512):
        """
        Initialize RemBERT model for skill classification
        
        Args:
            max_seq_length: Maximum sequence length for input texts
        """
        set_seed(42)
        
        # Initialize RemBERT model and tokenizer
        self.model_name = "google/rembert"
        self.tokenizer = RemBertTokenizer.from_pretrained(self.model_name)
        self.model = RemBertModel.from_pretrained(self.model_name)
        
        # Create SentenceTransformer wrapper
        self.sbert_model = SentenceTransformer(self.model_name)
        self.sbert_model.max_seq_length = max_seq_length

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 32,
                   num_epochs: int = 5,
                   lr: float = 2e-5,
                   warmup_steps: int = None):
        """
        Train model using MultipleNegativesRankingLoss and evaluate with ESCO job matching
        
        Args:
            pos_pairs: List of positive pairs [(text1, text2), ...]
            neg_pairs: List of negative pairs [(text1, text2), ...]
            output_path: Path to save model and evaluation results
            batch_size: Training batch size 
            num_epochs: Number of training epochs
            lr: Learning rate
            warmup_steps: Number of warmup steps for learning rate scheduler
            
        Returns:
            Tuple of (MRR results, best model)
        """
        # Initialize tracking
        MRR = []
        MRR_AT = 100
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        max_MRR = 0
        best_model_to_save = None
        
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        
        # Create train/dev split
        pos_train_samples, pos_dev_samples = train_test_split(pos_pairs, test_size=0.2, random_state=42)
        print(f"Training samples: {len(pos_train_samples)}, Development samples: {len(pos_dev_samples)}")
        
        # Create development set for evaluation
        dev_set_total = create_training_samples(pos_dev_samples, neg_pairs)
        
        # Prepare training data
        train_examples = []
        for item in pos_train_samples:
            train_examples.append(InputExample(texts=[item[0], item[1]]))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps if not provided (10% of training data)
        if warmup_steps is None:
            warmup_steps = int(len(pos_train_samples) * 0.1)
        
        # Initialize loss and evaluator
        train_loss = losses.MultipleNegativesRankingLoss(self.sbert_model)
        evaluator = evaluation.RerankingEvaluator(dev_set_total, at_k=100, show_progress_bar=True)
        
        # Train the model
        print("Training model...")
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
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
        
        # Calculate similarities and MRR for each embedding type
        similarities = {}
        for k in embeddings:
            print(f"\nEvaluating {k} embeddings...")
            similarities[k] = util.cos_sim(testads["embeddings_short"], embeddings[k]["embeddings"])
            
            ranks = []
            missing = 0
            simdf = pd.DataFrame(similarities[k], columns=embeddings[k]["esco_id"], index=testads["esco_id"])
            
            for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
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
                "training_details": [training_start, batch_size, lr, warmup_steps, num_epochs]
            }
            MRR.append(current_run)
            
            df = pd.DataFrame(MRR)
            print(f"\nResults for {k}:")
            print(df)
            
            # Save if best model
            if np.mean(ranks) > max_MRR:
                print(f"New best model saved (MRR: {np.mean(ranks):.4f})")
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
    
    print(f"Training with {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
    
    # Train model
    model = RemBERTSkillClassifier()
    mrr_results, best_model = model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/SBERT_Models/models/rembert_skill_classifier",
        batch_size=32,
        num_epochs=5,
        lr=2e-5
    )
    
    # Print final results
    print("\nTraining completed")
    df = pd.DataFrame(mrr_results)
    print("\nFinal Results:")
    print(df)