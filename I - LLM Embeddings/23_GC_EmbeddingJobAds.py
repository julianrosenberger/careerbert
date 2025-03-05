from openai import OpenAI
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from helpers import *
import torch
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Define OpenAI models
openai_models = {
    "openai-3-small": "text-embedding-3-small"
}

def process_model(model_name: str, model_path: str):
    """Process a single model's embeddings"""
    print(f"\nProcessing {model_name}")
    
    base_path = f"../00_data/SBERT_Models/models/{model_path}"
    
    # Check if embeddings already exist
    embedding_path = f"{base_path}/embeddings.pkl"
    if os.path.exists(embedding_path):
        print(f"Loading saved OpenAI embeddings for {model_path}")
        embeddings = load_pickle(embedding_path)
    else:
        print(f"Generating new OpenAI embeddings for {model_path}")
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        embeddings = encode_jobs_openai(
            client,
            save_path=base_path,
            model=model_name
        )

    print("\nCreating encodings for ads...")
    df_new = filter_communities_total(
        ads_df,
        "short+title",
        client,
        model_name,
        min_community_size=100,
        threshold=0.5
    )

    # Create centroids
    print("\nCreating centroids...")
    uniqueids = set(df_new["esco_id"].unique())
    unfiltered_centroids_dict = {}
    filtered_centroids_dict = {}

    for id in tqdm(uniqueids, desc="Processing job IDs"):
        # Creates the unfiltered centroids
        id_filter = df_new[df_new["esco_id"]==id]
        unfiltered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
        unfiltered_centroids_dict[id] = unfiltered_centroids

        # Creates the filtered centroids
        id_filter = df_new[(df_new["esco_id"]==id) & (df_new["community"]==0)]
        filtered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
        filtered_centroids_dict[id] = filtered_centroids

    # Save centroids
    write_pickle(f"{base_path}/centroids_unfiltered.pkl", unfiltered_centroids_dict)
    write_pickle(f"{base_path}/centroids_filtered.pkl", filtered_centroids_dict)

    # Update embeddings with centroids
    print("\nUpdating embeddings with centroids...")
    for centroid_path in ["centroids_filtered.pkl", "centroids_unfiltered.pkl"]:
        centroids = load_pickle(f"{base_path}/{centroid_path}")
        centroids_ids = list(centroids.keys())
        centroid_values = list(centroids.values())
        centroid_jobs = [esco_lookup[x] for x in centroids_ids]
        centroid_kind = "adcentroid_" + centroid_path.split("_")[1].split(".")[0]
        embeddings[centroid_kind] = {
            "jobtitle": centroid_jobs,
            "esco_id": centroids_ids,
            "embeddings": centroid_values
        }

    # Create job centroids
    print("\nCreating job centroids...")
    job_centroid = []
    for k in ['desc', 'adcentroid_unfiltered']:
        for id, job, embedding in zip(embeddings[k]["esco_id"],
                                    embeddings[k]["jobtitle"],
                                    embeddings[k]["embeddings"]):
            job_centroid.append({
                "esco_id": id,
                "jobtitle": job,
                "embeddings": embedding
            })

    # Save final mappings
    print("\nSaving final mappings...")
    job_embeddings_jobtitle, job_embeddings_esco_ids, job_embeddings = [], [], []
    centroid_df = pd.DataFrame(job_centroid)
    
    for id in tqdm(centroid_df["esco_id"].unique(), desc="Processing unique jobs"):
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

    write_pickle(f"{base_path}/embeddings.pkl", embeddings)
    print(f"Completed processing for {model_name}")
    
    # Save summary statistics
    stats = {
        'timestamp': str(datetime.now()),
        'n_jobs': len(uniqueids),
        'n_ads': len(df_new),
        'model': model_name
    }
    
    with open(f"{base_path}/processing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    # Load ads data
    print("Loading ads data...")
    global ads_df
    ads_df = pd.DataFrame(load_json("../00_data/EURES/0_pars_short_ads_final.json"))
    ads_df["short+title"] = ads_df.apply(concat_short, axis=1)

    # Load ESCO jobs for lookup
    global jobs, esco_lookup
    jobs = load_json("../00_data/ESCO/ESCO_JOBS_ALL.json")
    esco_lookup = {job["jobid_esco"]: job["jobtitle"] for job in jobs}
    
    for model_path, model_name in openai_models.items():
        process_model(model_name, model_path)

if __name__ == "__main__":
    main()