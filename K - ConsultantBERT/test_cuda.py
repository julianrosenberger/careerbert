import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
from sentence_transformers import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from helpers import *

class ConsultantBERT:
    """
    Implementation following Figure 2 architecture from ConsultantBERT paper,
    but using MNR loss instead of classification/regression objectives.
    """
    def __init__(
        self, 
        max_seq_length: int = 512,
        model_name: str = "bert-base-multilingual-cased"
    ):
        # Create base BERT model with correct components as in paper's Figure 2
        word_embedding_model = models.Transformer('google-bert/bert-base-multilingual-cased', max_seq_length=max_seq_length)
        
        # Add mean pooling as shown in paper's architecture
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,  # Paper uses mean pooling, not CLS
            pooling_mode_max_tokens=False
        )
        
        # Create bi-encoder SBERT model
        self.model = SentenceTransformer(modules=[
            word_embedding_model,
            pooling_model,
        ])
        
        self.max_seq_length = max_seq_length
        
    def prepare_training_data(
        self,
        pos_pairs: List[Tuple[str, str]]
    ) -> List[InputExample]:
        """Prepare training data from positive pairs for MNR loss"""
        train_examples = []
        
        # For MNR loss, we only need pairs without explicit labels
        for resume, vacancy in pos_pairs:
            train_examples.append(InputExample(texts=[resume, vacancy]))
                
        return train_examples
    
    def train(
        self,
        train_examples: List[InputExample],
        validation_examples: List[InputExample] = None,
        batch_size: int = 32,  # Same as your notebook for comparison
        epochs: int = 5,       # Paper uses 5 epochs
        warmup_steps: int = None,
        evaluation_steps: int = 1000,
        output_path: str = None
    ):
        """Train the model with MNR loss instead of paper's objectives"""
        
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size,
            pin_memory=True
        )
        
        # Use MNR loss instead of paper's classification/regression
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calculate warmup steps (10% of total steps)
        if warmup_steps is None:
            warmup_steps = int(len(train_examples) * 0.1)
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=None,  # We'll evaluate separately
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': 2e-5},
            show_progress_bar=True,
            output_path=output_path
        )
    
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device='cuda',
            normalize_embeddings=True
        )
    
    def predict(self, resumes: List[str], vacancies: List[str]) -> np.ndarray:
        """Predict similarity scores between resumes and vacancies"""
        resume_embeddings = self.encode(resumes)
        vacancy_embeddings = self.encode(vacancies)
        return util.cos_sim(resume_embeddings, vacancy_embeddings).cpu().numpy()
    
    def save(self, path: str):
        """Save the model"""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model"""
        model = cls()
        model.model = SentenceTransformer(path)
        return model

def main():
    # Load data
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    
    # Split data 80/10/10 as in paper
    train_pos, val_test = train_test_split(pos_pairs, test_size=0.2, random_state=42)
    val_pos, test_pos = train_test_split(val_test, test_size=0.5, random_state=42)
    
    # Initialize model
    model = ConsultantBERT()
    
    # Prepare training data
    train_examples = model.prepare_training_data(train_pos)
    val_examples = model.prepare_training_data(val_pos)
    
    # Train model
    output_path = "../00_data/SBERT_Models/consultantbert_mnr"
    model.train(
        train_examples=train_examples,
        validation_examples=val_examples,
        batch_size=32,
        epochs=5,
        output_path=output_path
    )
    
    # Evaluate on test ads
    print("\nEvaluating model...")
    testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
    encodings = model.encode(list(testads["short_texts"]))
    testads["embeddings"] = encodings.tolist()
    
    print("\nTraining and evaluation completed!")
    return model

if __name__ == "__main__":
    main()