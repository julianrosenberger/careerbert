from sentence_transformers import util
from datetime import datetime
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from helpers import *
import torch
from openai import OpenAI

# Set random seed for reproducibility
random.seed(42)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Define models to evaluate
models = {
    "openai-3-small": "text-embedding-3-small"
}

def encode_test_ads(testads, client, model_name):
    """Encode test ads using OpenAI API"""
    print(f"Encoding test ads with {model_name}")
    encodings = []
    
    batch_size = 20
    for i in tqdm(range(0, len(testads), batch_size)):
        batch = testads[i:i + batch_size]
        batch_encodings = [
            get_embedding(text, client, model_name) 
            for text in batch["short_texts"]
        ]
        encodings.extend(batch_encodings)
    
    return np.array(encodings)

def evaluate_embeddings(model_path, model_name=None):
    """Evaluate embeddings for a specific model"""
    print(f"\nEvaluating embeddings for {model_path}")
    
    # Load test ads
    testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
    
    # Load embeddings
    embedding_path = f"../00_data/SBERT_Models/models/{model_path}/embeddings.pkl"
    
    if not os.path.exists(embedding_path):
        print(f"No embeddings found at {embedding_path}")
        return None
        
    embeddings = load_pickle(embedding_path)
    print(f"Found embeddings with keys: {embeddings.keys()}")
    
    # Check for existing test embeddings or generate new ones
    test_embedding_path = f"../00_data/SBERT_Models/models/{model_path}/test_embeddings.pkl"
    if os.path.exists(test_embedding_path):
        print("Loading existing test embeddings")
        encodings_short = load_pickle(test_embedding_path)
    else:
        print("Generating new test embeddings")
        encodings_short = encode_test_ads(testads, client, model_name)
        write_pickle(test_embedding_path, encodings_short)
        
    testads["embeddings_short"] = encodings_short.tolist()

    # Initialize results
    MRR = []
    MRR_AT = 100
    currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])

    # Evaluate embeddings
    for textkind in ["embeddings_short"]:
        similarities = {}
        
        for k in ["job_centroid"]:
            test_embeddings = torch.tensor(testads[textkind].tolist())
            job_embeddings = torch.tensor(embeddings[k]["embeddings"])
            
            similarities[k] = util.cos_sim(test_embeddings, job_embeddings)
        
        for k in similarities:
            ranks = []
            missing = 0
            max_similarity = max(map(max, similarities[k]))
            
            simdf = pd.DataFrame(
                similarities[k],
                columns=embeddings[k]["esco_id"],
                index=testads["esco_id"]
            )
            
            for i in tqdm(range(len(simdf)), desc="Processing ranks"):
                id = simdf.iloc[i].name
                series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                rank = (series[series["index"] == id].index.item() + 1)
                
                if rank > MRR_AT:
                    missing += 1
                    ranks.append(0)
                else:
                    ranks.append(1/rank)
            
            missing = missing / len(simdf)
            
            MRR.append({
                "model": model_name if model_name else model_path.split("/")[-2],
                "textkind": textkind,
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "max_similarity": max_similarity.item(),
                "MRR@": MRR_AT
            })
    
    # Create and save results
    df = pd.DataFrame(MRR).sort_values(by=["MRR"], ascending=[False]).reset_index(drop=True)
    
    output_dir = "../00_data/SBERT_Models/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{currently}_evaluation_llm.xlsx"
    df.to_excel(output_path)
    
    print(f"\nResults saved to {output_path}")
    print("\nEvaluation Results:")
    print(df)
    
    return df

def main():
    all_results = []
    
    for model_path, model_name in models.items():
        df = evaluate_embeddings(model_path, model_name=model_name)
        if df is not None:
            all_results.append(df)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        output_path = f"../00_data/SBERT_Models/evaluation/{currently}_evaluation_combined.xlsx"
        final_df.to_excel(output_path)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()