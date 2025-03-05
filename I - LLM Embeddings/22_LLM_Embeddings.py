from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import os
import random
import pandas as pd
from helpers import *
import numpy as np
from tqdm import tqdm
from openai import OpenAI

random.seed(42)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# Define OpenAI models
openai_models = {
    "openai-ada": "text-embedding-ada-002",
    "openai-3-small": "text-embedding-3-small",
    "openai-3-large": "text-embedding-3-large"
}

testads = pd.DataFrame(load_json(r"../00_data/EURES/eures_testads_final_short.json"))

# Add base models to the evaluation
paths = [
        "openai-3-large",
        "jobgbert_batch32_woTSDAE_2e-05_f10/",  
        "openai-3-small",
        "openai-ada",
        "gbert_batch32_woTSDAE_2e-05_f10/",
        "consultantbert_multilingual_regressor/",
        "deepset/gbert-base",        # Base GBERT model
        "agne/jobGBERT",            # Base JobGBERT model
        "google/rembert",            # Base RemBERT model
]

# Evaluate with Test Ads
MRR = []
MRR_AT = 100
currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])

for model_path in paths:
    print(f"Loading Model {model_path}")
    
    if model_path in ["agne/jobGBERT", "deepset/gbert-base", "google/rembert"]:
        model = SentenceTransformer(model_path)
        embeddings = encode_jobs(model)
    elif model_path in openai_models:
        # Check if embeddings already exist
        embedding_path = f"../00_data/SBERT_Models/models/{model_path}/embeddings.pkl"
        if os.path.exists(embedding_path):
            print(f"Loading saved OpenAI embeddings for {model_path}")
            embeddings = load_pickle(embedding_path)
        else:
            print(f"Generating new OpenAI embeddings for {model_path}")
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            embeddings = encode_jobs_openai(
                client, 
                save_path=f"../00_data/SBERT_Models/models/{model_path}",
                model=openai_models[model_path]
            )
    else:
        model = SentenceTransformer(f"../00_data/SBERT_Models/models/{model_path}")
        embeddings = load_pickle(f"../00_data/SBERT_Models/models/{model_path}/embeddings.pkl")

    print("Creating Embeddings for test ads.")
    if model_path in openai_models:
        # Check if test embeddings exist
        test_embedding_path = f"../00_data/SBERT_Models/models/{model_path}/test_embeddings.pkl"
        if os.path.exists(test_embedding_path):
            print(f"Loading saved test embeddings for {model_path}")
            encodings_short = load_pickle(test_embedding_path)
        else:
            print(f"Generating new test embeddings for {model_path}")
            os.makedirs(os.path.dirname(test_embedding_path), exist_ok=True)
            encodings_short = encode_ads_openai(
                client, 
                list(testads["short_texts"]), 
                model=openai_models[model_path]
            )
            write_pickle(test_embedding_path, encodings_short)
    else:
        encodings_short = model.encode(list(testads["short_texts"]), show_progress_bar=True)

    testads["embeddings_short"] = encodings_short.tolist()

    print("Finished creating Embeddings. Evaluating.")

    for textkind in ["embeddings_short"]:
        similarities = {}
        for k in ["desc"]:
            similarities[k] = (util.cos_sim(testads[textkind], embeddings[k]["embeddings"]))

        for k in similarities:
            ranks = []
            missing = 0
            max_similarity = (max(map(max, similarities[k])))
            simdf = pd.DataFrame(similarities[k], columns=embeddings[k]["esco_id"], index=testads["esco_id"])
            for i in tqdm(range(len(simdf))):
                id = simdf.iloc[i].name
                series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                rank = (series[series["index"] == id].index.item() + 1)
                if rank > MRR_AT:
                    missing += 1
                    ranks.append(0)
                else:
                    ranks.append(1 / rank)
            missing = missing / len(simdf)
            
            # Add a flag to indicate if it's a base model
            is_base = model_path in ["agne/jobGBERT", "deepset/gbert-base"]
            model_type = "base" if is_base else "fine-tuned" if "gbert" in model_path.lower() else "pretrained"
            
            MRR.append({
                "model": model_path,
                "textkind": textkind,
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "max_similarity": max_similarity,
                "MRR@": MRR_AT,
                "model_type": model_type,
                "embedding_model": openai_models.get(model_path, model_path)
            })
            
            # Create DataFrame and sort, grouping base and fine-tuned versions together
            df = pd.DataFrame(MRR)
            df['sort_key'] = df['model'].apply(lambda x: x.lower().replace('base', '0'))  # Make base models appear first
            df = df.sort_values(by=["sort_key", "MRR"], ascending=[True, False]).reset_index(drop=True)
            df = df.drop('sort_key', axis=1)
            
            print("\nCurrent Results:")
            print(df)
            
            # Save results
            df.to_excel(f"../00_data/SBERT_Models/evaluation/{currently}_evaluation_with_base.xlsx")