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
    #api_key="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",
    api_key=OPENAI_API_KEY
)

# Define OpenAI models
openai_models = {
    "openai-ada": "text-embedding-ada-002",
    "openai-3-small": "text-embedding-3-small",
    #"openai-3-large": "text-embedding-3-large"
}

testads = pd.DataFrame(load_json(r"../00_data/EURES/eures_testads_final_short.json"))
ads_df = pd.DataFrame(load_json(r"../00_data/EURES/0_pars_short_ads_final.json"))
ads_df["short+title"] = ads_df.apply(concat_short, axis=1)

jobs = (load_json("../00_data/ESCO/ESCO_JOBS_ALL.json"))
esco_lookup = {}
for job in jobs:
  esco_lookup[job["jobid_esco"]] = job["jobtitle"]


paths = [
        "openai-ada",
        "openai-3-small", 
        "openai-3-large",
        # "deepset/gbert-base",
        #   "agne/jobGBERT",
        #   "jobgbert_TSDAE_epochs5/",
        #   "gbert_TSDAE_epochs5/",
        #   "jobgbert_batch16_woTSDAE_2e-05_f10/",
        #   "jobgbert_batch16_wTSDAE_2e-05_f10/",
        #   "jobgbert_batch32_woTSDAE_2e-05_f10/",  #****
        #   "jobgbert_batch32_wTSDAE_2e-05_f10/",
        #   "jobgbert_batch64_woTSDAE_2e-05_f10/",
        #   "jobgbert_batch64_wTSDAE_2e-05_f10/",
        #   "gbert_batch16_woTSDAE_2e-05_f10/",
        #   "gbert_batch16_wTSDAE_2e-05_f10/",
        #   "gbert_batch32_woTSDAE_2e-05_f10/",
        #  "gbert_batch32_wTSDAE_2e-05_f10/",
        #   "gbert_batch64_woTSDAE_2e-05_f10/",
        #   "gbert_batch64_wTSDAE_2e-05_f10/",
]


# Evaluate with Test Ads
MRR = []
MRR_AT = 100
currently = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])

for model_path in paths:
    print(f"Loading Model {model_path}")
    
    if model_path in ["agne/jobGBERT", "deepset/gbert-base"]:
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
    
    # Create centroids after loading embeddings
    if model_path in openai_models:
        print("Creating centroids...")
        # Load or create embeddings for all ads
        all_ads_embedding_path = f"../00_data/SBERT_Models/models/{model_path}/all_ads_embeddings.pkl"
        if os.path.exists(all_ads_embedding_path):
            print("Loading saved job ad embeddings")
            ads_embeddings = load_pickle(all_ads_embedding_path)
            ads_df["embeddings"] = ads_embeddings
        else:
            print("Creating new job ad embeddings")
            ads_embeddings = encode_ads_openai(
                client,
                list(ads_df["short+title"]),
                model=openai_models[model_path]
            )
            write_pickle(all_ads_embedding_path, ads_embeddings)
            ads_df["embeddings"] = list(ads_embeddings)

        # Create communities and centroids
        df_new = filter_communities_total(ads_df, "short+title", None, min_community_size=100, threshold=0.5)
        
        # Create centroids
        uniqueids = set(df_new["esco_id"].unique())
        unfiltered_centroids_dict = {}
        filtered_centroids_dict = {}
        
        print("Creating centroids...")
        for id in tqdm(uniqueids):
            # Creates the unfiltered centroids
            id_filter = df_new[df_new["esco_id"]==id]
            unfiltered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
            unfiltered_centroids_dict[id] = unfiltered_centroids

            # Creates the filtered centroids
            id_filter = df_new[(df_new["esco_id"]==id) & (df_new["community"]==0)]
            filtered_centroids = np.stack(list(id_filter["embeddings"])).mean(axis=0, dtype="float32")
            filtered_centroids_dict[id] = filtered_centroids

        # Add centroids to embeddings dictionary
        centroids_ids = list(unfiltered_centroids_dict.keys())
        centroid_values = list(unfiltered_centroids_dict.values())
        embeddings["job_centroid"] = {
            "jobtitle": [esco_lookup[x] for x in centroids_ids],
            "esco_id": centroids_ids,
            "embeddings": centroid_values
        }
        
        # Save updated embeddings
        write_pickle(f"{model_path}embeddings.pkl", embeddings)

    # Then continue with evaluation, but change k to "job_centroid":
    for textkind in ["embeddings_short"]:
        similarities = {}
        for k in ["job_centroid"]:  # Changed from "desc" to "job_centroid"
            similarities[k] = (util.cos_sim(testads[textkind], embeddings[k]["embeddings"]))
        

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
            
            MRR.append({
                "model": model_path,
                "textkind": textkind,
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "max_similarity": max_similarity,
                "MRR@": MRR_AT,
                "embedding_model": openai_models.get(model_path, model_path)  # Store actual model name for OpenAI
            })
            df = pd.DataFrame(MRR).sort_values(by=["MRR"], ascending=[False]).reset_index(drop=True)
            print(df)
            df.to_excel(f"../00_data/SBERT_Models/evaluation/{currently}_evaluation.xlsx")