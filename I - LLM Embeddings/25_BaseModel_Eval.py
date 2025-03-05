from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from helpers import load_json, write_pickle, load_pickle
from helpers import *

# Define OpenAI models
openai_models = {
    "openai-ada": "text-embedding-ada-002",
    "openai-3-small": "text-embedding-3-small",
    "openai-3-large": "text-embedding-3-large"
}

# Load test ads
testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))

# Initialize results list
results = []
currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])

# Define models to evaluate (use only base models)
paths = [
    "openai-3-large",
    "deepset/gbert-base",  # Base GBERT
    "openai-3-small",
    "openai-ada",
    "agne/jobGBERT",      # Base JobGBERT
]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for model_path in paths:
    print(f"Evaluating base model: {model_path}")
    
    # Get embeddings for test ads using base model
    if model_path in openai_models:
        # Use OpenAI embeddings
        encodings_short = encode_ads_openai(
            client, 
            list(testads["short_texts"]), 
            model=openai_models[model_path]
        )
    else:
        # Use base SBERT model directly
        model = SentenceTransformer(model_path)
        encodings_short = model.encode(list(testads["short_texts"]), show_progress_bar=True)
    
    # Calculate similarities between all test ads
    similarities = util.cos_sim(encodings_short, encodings_short)
    
    # For each unique ESCO ID
    avg_intra_similarities = []
    for esco_id in testads["esco_id"].unique():
        # Get indices of ads with this ESCO ID
        indices = testads.index[testads["esco_id"] == esco_id].tolist()
        
        if len(indices) > 1:  # Need at least 2 ads to compute similarity
            # Get similarities between all ads with this ESCO ID
            intra_sims = similarities[indices][:, indices]
            
            # Remove self-similarities (diagonal)
            mask = ~np.eye(len(indices), dtype=bool)
            avg_sim = intra_sims[mask].mean()
            avg_intra_similarities.append(avg_sim)
    
    # Store results
    results.append({
        "model": model_path,
        "avg_similarity": np.mean(avg_intra_similarities),
        "std_similarity": np.std(avg_intra_similarities),
        "embedding_model": openai_models.get(model_path, model_path)
    })
    
    # Print current results
    df = pd.DataFrame(results).sort_values(by=["avg_similarity"], ascending=[False])
    print("\nCurrent Results:")
    print(df)

# Save final results
df.to_excel(f"../00_data/SBERT_Models/evaluation/{currently}_base_model_evaluation.xlsx")