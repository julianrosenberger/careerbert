from datetime import datetime
from gensim.models import Word2Vec, Doc2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from helpers import *
import os

class DenseModelEvaluator:
    """Evaluator for Word2Vec and Doc2Vec models using test ads"""
    
    def __init__(self):
        self.testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
        self.MRR_AT = 100
        
    def load_model(self, model_path: str, model_type: str):
        """Load Word2Vec or Doc2Vec model"""
        if model_type == "word2vec":
            return Word2Vec.load(model_path)
        else:
            return Doc2Vec.load(model_path)
            
    def get_document_vector(self, text: str, model, model_type: str) -> np.ndarray:
        """Get vector representation for a document"""
        # Preprocess text (same as training)
        text = text.lower()
        text = "".join([x for x in text if x.isalnum() or x.isspace()])
        tokens = text.split()
        
        if model_type == "word2vec":
            vectors = []
            for token in tokens:
                if token in model.wv:
                    vectors.append(model.wv[token])
            if vectors:
                return np.mean(vectors, axis=0)
            return np.zeros(model.vector_size)
        else:
            return model.infer_vector(tokens)
            
    def compute_similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity with consistent dtype handling"""
        embeddings1 = torch.tensor(embeddings1, dtype=torch.float32)
        embeddings2 = torch.tensor(embeddings2, dtype=torch.float32)
        
        # Normalize
        embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        return torch.mm(embeddings1, embeddings2.t()).numpy()
        
    def evaluate_model(self, model_path: str, model_type: str):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_type} model from: {model_path}")
        
        # Load model
        model = self.load_model(model_path, model_type)
        
        # Get embeddings from test ads
        print("Creating test ad embeddings...")
        test_embeddings = []
        for text in tqdm(self.testads["short_texts"]):
            test_embeddings.append(self.get_document_vector(text, model, model_type))
        test_embeddings = np.array(test_embeddings)
        
        # Load job embeddings
        embeddings = load_pickle(f"{os.path.dirname(model_path)}/embeddings.pkl")
        
        # Initialize results tracking
        MRR = []
        currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        
        # Calculate similarities and evaluate
        print("Evaluating...")
        for k in ["job_centroid"]:  # Using same evaluation type as BERT
            similarities = self.compute_similarity(test_embeddings, embeddings[k]["embeddings"])
            
            ranks = []
            missing = 0
            simdf = pd.DataFrame(similarities, 
                               columns=embeddings[k]["esco_id"], 
                               index=self.testads["esco_id"])
            
            for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
                id = simdf.iloc[i].name
                series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                rank = (series[series["index"]==id].index.item()+1)
                
                if rank > self.MRR_AT:
                    missing += 1
                    ranks.append(0)
                else:
                    ranks.append(1/rank)
            
            missing = missing/len(simdf)
            current_run = {
                "model": model_path.split("/")[-2],
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "MRR@": self.MRR_AT,
                "max_similarity": similarities.max()
            }
            MRR.append(current_run)
            
            # Display and save results
            df = pd.DataFrame(MRR)
            print("\nResults:")
            print(df)
            df.to_excel(f"../00_data/Dense_Models/evaluation/{currently}_evaluation.xlsx")
            
        return MRR

def main():
    """Main evaluation function"""
    evaluator = DenseModelEvaluator()
    
    # Define models to evaluate
    models = [
        ("../00_data/Dense_Models/doc2vec_dbow_best/doc2vec-dbow.model", "doc2vec"),
        ("../00_data/Dense_Models/word2vec_best/word2vec.model", "word2vec"),
        ("../00_data/Dense_Models/doc2vec_dm_best/doc2vec-dm.model", "doc2vec")
    ]
    
    # Track all results
    all_results = []
    
    # Evaluate each model
    for model_path, model_type in models:
        if os.path.exists(model_path):
            try:
                results = evaluator.evaluate_model(model_path, model_type)
                all_results.extend(results)
            except Exception as e:
                print(f"Error evaluating {model_path}: {str(e)}")
        else:
            print(f"Model not found: {model_path}")
    
    # Save combined results
    if all_results:
        df = pd.DataFrame(all_results)
        currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        df.to_excel(f"../00_data/Dense_Models/evaluation/{currently}_combined_evaluation.xlsx")
        print("\nFinal Combined Results:")
        print(df)

if __name__ == "__main__":
    main()