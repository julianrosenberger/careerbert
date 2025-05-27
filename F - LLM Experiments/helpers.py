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

def concat_short(ad):
    return ad["title"] + ad["short_texts"]

# =============================================================================
# OPENAI EMBEDDINGS
# =============================================================================

def get_embedding(text, client, model="text-embedding-ada-002"):
    """Get embedding for a single text using specified OpenAI model"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

def encode_ads_openai(client, ads, model="text-embedding-ada-002"):
    """
    Encode test ads using OpenAI embeddings
    Args:
        client: OpenAI client instance
        ads: List of ad texts to encode
        model: OpenAI model to use for embeddings
    Returns:
        numpy array of embeddings
    """
    print(f"Encoding ads using {model}")
    ads_embeddings = np.array(list(tqdm(map(
        lambda x: get_embedding(x, client, model), ads
    ))), dtype=np.float32)
    return ads_embeddings

def encode_jobs_openai(client, save_path=None, model="text-embedding-ada-002"):
    """
    Encode jobs using OpenAI embeddings and optionally save them
    Args:
        client: OpenAI client instance
        save_path: Optional path to save the embeddings
        model: OpenAI model to use for embeddings
    Returns:
        Dictionary containing the embeddings
    """
    jobs = load_json("../00_data/ESCO/ESCO_JOBS_ALL.json")
    embedding_dict = {x: {} for x in ["skillsets", "desc", "jobtitle"]}
    jobtitles = [x["jobtitle"] for x in jobs]
    skillsets = ([" ".join(x["full_skills"]) for x in jobs])
    descs = [x["jobdescription"] for x in jobs]
    escoids = [x["jobid_esco"] for x in jobs]

    print(f"Generating embeddings using {model}")
    skill_embeddings = np.array(list(tqdm(map(
        lambda x: get_embedding(x, client, model), skillsets
    ))), dtype=np.float32)
    desc_embeddings = np.array(list(tqdm(map(
        lambda x: get_embedding(x, client, model), descs
    ))), dtype=np.float32)
    title_embeddings = np.array(list(tqdm(map(
        lambda x: get_embedding(x, client, model), jobtitles
    ))), dtype=np.float32)

    embedding_dict["skillsets"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": skill_embeddings})
    embedding_dict["desc"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": desc_embeddings})
    embedding_dict["jobtitle"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": title_embeddings})

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save embeddings
        embedding_path = os.path.join(save_path, "embeddings.pkl")
        write_pickle(embedding_path, embedding_dict)
        
        # Save model info
        model_info = {
            "model": model,
            "embedding_dimensions": len(skill_embeddings[0])
        }
        write_json(os.path.join(save_path, "model_info.json"), model_info)
        
        print(f"Embeddings saved to {embedding_path}")

    return embedding_dict

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
# UNIFIED MODEL HANDLING
# =============================================================================

def encode_texts(texts, model_info, show_progress=True):
    """Unified text encoding"""
    if model_info["type"] == "openai":
        return encode_ads_openai(model_info["client"], texts, model_info["model"])
    else:
        return model_info["model"].encode(texts, show_progress_bar=show_progress)

def create_model_and_get_embeddings(model_path, force_regenerate=False):
    """Unified model creation and embedding loading/generation"""
    config = Config()
    
    if model_path in config.OPENAI_MODELS:
        # OpenAI model handling
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        embedding_path = f"{config.MODELS_BASE}/{model_path}/embeddings.pkl"
        if os.path.exists(embedding_path) and not force_regenerate:
            print(f"Loading saved embeddings for {model_path}")
            embeddings = load_pickle(embedding_path)
        else:
            print(f"Generating embeddings for {model_path}")
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            embeddings = encode_jobs_openai(
                client, 
                save_path=f"{config.MODELS_BASE}/{model_path}",
                model=config.OPENAI_MODELS[model_path]
            )
        
        model_info = {"type": "openai", "model": config.OPENAI_MODELS[model_path], "client": client}
        
    elif model_path in ["agne/jobGBERT", "deepset/gbert-base", "google/rembert"]:
        # Base models
        model = SentenceTransformer(model_path)
        embeddings = encode_jobs(model)
        model_info = {"type": "sbert", "model": model}
        
    else:
        # Custom trained models
        model = SentenceTransformer(f"{config.MODELS_BASE}/{model_path}")
        embeddings = load_pickle(f"{config.MODELS_BASE}/{model_path}/embeddings.pkl")
        model_info = {"type": "sbert", "model": model}
    
    return model_info, embeddings

# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def community_detection(embeddings, threshold=0.75, min_community_size=10, batch_size=1024):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores
        cos_scores = util.cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)
        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                # Check if we need to increase sort_max_size
                while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities

def filter_communities_total(df, columnname_to_encode, client, model, min_community_size=10, threshold=0.75):
    """
    Modified version of filter_communities_total to work with OpenAI embeddings
    Args:
        df: DataFrame containing the ads
        columnname_to_encode: Column name containing text to encode
        client: OpenAI client
        model: OpenAI model name
        min_community_size: Minimum size for communities
        threshold: Similarity threshold for community detection
    Returns:
        DataFrame with communities and embeddings
    """
    unique_texts = list(set(df[columnname_to_encode]))
    print(f"{len(unique_texts)} ads to encode.")
    
    # Create embeddings using OpenAI
    embeddings = np.array([get_embedding(text, client, model) for text in tqdm(unique_texts)], 
                         dtype=np.float32)
    
    # Create mapping from text to embedding
    embedding_map = {text: emb for text, emb in zip(unique_texts, embeddings)}
    df["embeddings"] = df[columnname_to_encode].map(embedding_map)
    
    results = []
    esco_ids = list(df["esco_id"].unique())
    df["community"] = None
    print("Adding Communities.")
    
    for id in tqdm(esco_ids):
        filtered_df = df[df["esco_id"]==id].reset_index(drop=True)
        
        if len(filtered_df) < min_community_size:
            filtered_df["community"] = 0
        else:
            # Convert numpy arrays to torch tensors
            embds = torch.tensor([x for x in filtered_df["embeddings"]], dtype=torch.float32)
            
            communities = community_detection(embds, threshold=threshold, 
                                           min_community_size=min_community_size)
            
            if len(communities) > 0:
                for index, community in enumerate(communities):
                    filtered_df.loc[community, "community"] = index
            else:
                filtered_df["community"] = 0
                
        results += filtered_df.to_dict("records")
        
    result_df = pd.DataFrame(results)
    result_df.reset_index(drop=True, inplace=True)
    return result_df

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