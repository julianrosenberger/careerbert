import json
import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Model configurations
    OPENAI_MODELS = {
        "openai-ada": "text-embedding-ada-002",
        "openai-3-small": "text-embedding-3-small",
        "openai-3-large": "text-embedding-3-large"
    }
    
    # Evaluation settings
    MRR_AT = 100
    BATCH_SIZE = 50
    MIN_COMMUNITY_SIZE = 100
    THRESHOLD = 0.75
    
    # Paths
    DATA_BASE = "../00_data"
    MODELS_BASE = f"{DATA_BASE}/SBERT_Models/models"
    EVALUATION_OUTPUT = f"{DATA_BASE}/SBERT_Models/evaluation"
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def write_json(filename, data):
    with open(filename, 'w', encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print("Successfully saved file:", filename)
    return filename

def load_pickle(filepath):
    with open(filepath, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data

def write_pickle(filepath, data):
    with open(filepath, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Pickle saved.")

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def flatten_list(list_of_lists):
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return flattened_list

def load_data_pairs():
    path_to_data = "../00_data/SBERT_Models/trainingdata/"
    traindata = {}
    for file in tqdm(os.listdir(path_to_data)):
        if "_pairs" in file:
            traindata[file.split(".")[0]] = load_json(path_to_data + file)
    return traindata

def load_standard_datasets():
    """Load commonly used datasets"""
    return {
        "esco_jobs": load_json("../00_data/ESCO/ESCO_JOBS_ALL.json"),
        "testads": pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json")),
        "ads_df": pd.DataFrame(load_json("../00_data/EURES/0_pars_short_ads_final.json"))
    }

# =============================================================================
# SENTENCE TRANSFORMER EMBEDDINGS
# =============================================================================

def encode_jobs(model):
    jobs = load_json("../00_data/ESCO/ESCO_JOBS_ALL.json")
    embedding_dict = {x: {} for x in ["skillsets", "desc", "jobtitle"]}
    jobtitles = [x["jobtitle"] for x in jobs]
    skillsets = ([" ".join(x["full_skills"]) for x in jobs])
    descs = [x["jobdescription"] for x in jobs]
    escoids = [x["jobid_esco"] for x in jobs]

    skill_embeddings = model.encode(skillsets, show_progress_bar=True)
    desc_embeddings = model.encode(descs, show_progress_bar=True)
    title_embeddings = model.encode(jobtitles, show_progress_bar=True)

    embedding_dict["skillsets"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": skill_embeddings})
    embedding_dict["desc"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": desc_embeddings})
    embedding_dict["jobtitle"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": title_embeddings})

    return embedding_dict

# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_embeddings(testads, embeddings, model_path):
    """
    Evaluate embeddings using MRR metric
    Args:
        testads: DataFrame with test ads and their embeddings
        embeddings: Dictionary containing job embeddings 
        model_path: String identifier for the model
    Returns:
        List of evaluation results
    """
    config = Config()
    results = []
    
    for textkind in ["embeddings_short"]:
        similarities = {}
        for k in ["desc"]:
            similarities[k] = util.cos_sim(testads[textkind], embeddings[k]["embeddings"])

        for k in similarities:
            ranks = []
            missing = 0
            max_similarity = max(map(max, similarities[k]))
            simdf = pd.DataFrame(similarities[k], columns=embeddings[k]["esco_id"], index=testads["esco_id"])
            
            for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
                id = simdf.iloc[i].name
                series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                rank = (series[series["index"] == id].index.item() + 1)
                if rank > config.MRR_AT:
                    missing += 1
                    ranks.append(0)
                else:
                    ranks.append(1 / rank)
            missing = missing / len(simdf)
            
            # Determine model type for classification
            if model_path in config.OPENAI_MODELS:
                model_type = "openai"
                embedding_model = config.OPENAI_MODELS[model_path]
            elif model_path in ["agne/jobGBERT", "deepset/gbert-base", "google/rembert"]:
                model_type = "base"
                embedding_model = model_path
            else:
                model_type = "fine-tuned"
                embedding_model = model_path
            
            results.append({
                "model": model_path,
                "textkind": textkind,
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "max_similarity": max_similarity,
                "MRR@": config.MRR_AT,
                "model_type": model_type,
                "embedding_model": embedding_model
            })
    
    return results

# =============================================================================
# CACHING UTILITIES
# =============================================================================

def get_or_create_embeddings(cache_path, create_func, force_regenerate=False):
    """Simple caching helper"""
    if os.path.exists(cache_path) and not force_regenerate:
        print(f"Loading cached embeddings from {cache_path}")
        return load_pickle(cache_path)
    else:
        print(f"Creating new embeddings, will save to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        embeddings = create_func()
        write_pickle(cache_path, embeddings)
        return embeddings