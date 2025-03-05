from sentence_transformers import util
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import json
import pandas as pd
import nltk
from collections import Counter
import random
import os
from datetime import datetime
import torch
from helpers import *
from gensim.models import Word2Vec, Doc2Vec
from Word2Vec_Training import ElsaftyDenseRecommender

# from sbert library, original function has a bug, changes are not merged. https://github.com/UKPLab/sentence-transformers/commit/d8982c9f0d44f8a3c41579fa64c603eca029649b
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

def filter_communities_total_w2v(df, columnname_to_encode, model, min_community_size=10, threshold=0.75):
    """
    Adapt community detection for Word2Vec/Doc2Vec models
    """
    unique_texts = list(set(df[columnname_to_encode]))
    print(f"{len(unique_texts)} ads to encode.")
    
    # Get embeddings using Word2Vec/Doc2Vec model
    embeddings = []
    for text in tqdm(unique_texts, desc="Encoding texts"):
        processed = model.preprocess_text(text)
        embeddings.append(model.get_document_vector(processed))
    
    # Create embedding map
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
            embds = [torch.FloatTensor(x) for x in filtered_df["embeddings"]]
            embds = torch.stack(embds)
            communities = community_detection(embds, threshold=threshold, min_community_size=min_community_size)
            if len(communities) > 0:
                for index, community in enumerate(communities):
                    filtered_df.loc[community, "community"] = index
            else:
                filtered_df["community"] = 0
        results += filtered_df.to_dict("records")
        
    result_df = pd.DataFrame(results)
    result_df.reset_index(drop=True, inplace=True)
    return result_df

def evaluate_dense_models():
    # Load job ads
    ads_df = pd.DataFrame(load_json("../00_data/EURES/0_pars_short_ads_final.json"))
    
    # Add title+short_text combination
    def concat_short(ad):
        return ad["title"] + ad["short_texts"]
    ads_df["short+title"] = ads_df.apply(concat_short, axis=1)
    
    # Load ESCO job lookup
    jobs = load_json("../00_data/ESCO/ESCO_JOBS_ALL.json")
    esco_lookup = {}
    for job in jobs:
        esco_lookup[job["jobid_esco"]] = job["jobtitle"]
    
    # Model paths to evaluate
    model_paths = [
        "../00_data/Dense_Models/doc2vec_dbow_best/doc2vec-dbow.model",
        "../00_data/Dense_Models/word2vec_best/word2vec.model",
        "../00_data/Dense_Models/doc2vec_dbow_best/doc2vec-dbow.model"
    ]
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        
        # Load appropriate model type
        if "word2vec" in model_path:
            model = Word2Vec.load(model_path)
            model_type = "word2vec"
        else:
            model = Doc2Vec.load(model_path)
            model_type = "doc2vec"
        
        # Create wrapper for consistent interface
        model_wrapper = ElsaftyDenseRecommender(model_type=model_type)
        model_wrapper.model = model
        
        # Generate embeddings and detect communities
        df_new = filter_communities_total_w2v(
            ads_df, 
            "short+title", 
            model_wrapper, 
            min_community_size=100,
            threshold=0.5
        )
        
        # Create centroids
        uniqueids = set(df_new["esco_id"].unique())
        unfiltered_centroids_dict = {}
        filtered_centroids_dict = {}
        
        print("Creating Centroids.")
        for id in tqdm(uniqueids):
            # Unfiltered centroids
            id_filter = df_new[df_new["esco_id"]==id]
            unfiltered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
            unfiltered_centroids_dict[id] = unfiltered_centroids
            
            # Filtered centroids
            id_filter = df_new[(df_new["esco_id"]==id) & (df_new["community"]==0)]
            filtered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
            filtered_centroids_dict[id] = filtered_centroids
        
        # Save centroids
        output_dir = os.path.dirname(model_path)
        write_pickle(f"{output_dir}/centroids_unfiltered.pkl", unfiltered_centroids_dict)
        write_pickle(f"{output_dir}/centroids_filtered.pkl", filtered_centroids_dict)
        
        # Generate and save embeddings in same format as BERT
        embeddings = encode_jobs(model_wrapper)  # Using your existing helper
        
        # Add centroid embeddings
        for centroid_path in ["centroids_filtered.pkl", "centroids_unfiltered.pkl"]:
            centroids = load_pickle(f"{output_dir}/{centroid_path}")
            centroid_ids = list(centroids.keys())
            centroid_values = list(centroids.values())
            centroid_jobs = [esco_lookup[x] for x in centroid_ids]
            centroid_kind = "adcentroid_" + centroid_path.split("_")[1].split(".")[0]
            embeddings[centroid_kind] = {
                "jobtitle": centroid_jobs,
                "esco_id": centroid_ids,
                "embeddings": centroid_values
            }
        
        # Create combined centroids
        job_centroid = []
        for k in ['desc', 'adcentroid_unfiltered']:
            for id, job, embedding in zip(
                embeddings[k]["esco_id"],
                embeddings[k]["jobtitle"], 
                embeddings[k]["embeddings"]
            ):
                job_centroid.append({
                    "esco_id": id,
                    "jobtitle": job,
                    "embeddings": embedding
                })
        
        # Calculate final embeddings
        job_embeddings_jobtitle, job_embeddings_esco_ids, job_embeddings = [], [], []
        centroid_df = pd.DataFrame(job_centroid)
        
        for id in centroid_df["esco_id"].unique():
            filtered_df = centroid_df[centroid_df["esco_id"]==id]
            stacked_embedding = np.stack(list(filtered_df["embeddings"])).mean(axis=0, dtype="float32")
            job_embeddings.append(stacked_embedding)
            job_embeddings_esco_ids.append(id)
            job_embeddings_jobtitle.append(filtered_df["jobtitle"].iloc[0])
        
        embeddings["job_centroid"] = {
            "jobtitle": job_embeddings_jobtitle,
            "esco_id": job_embeddings_esco_ids,
            "embeddings": job_embeddings
        }
        
        # Save final embeddings
        write_pickle(f"{output_dir}/embeddings.pkl", embeddings)

if __name__ == "__main__":
    evaluate_dense_models()