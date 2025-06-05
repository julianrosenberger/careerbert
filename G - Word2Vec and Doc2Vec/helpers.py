import json
import os
from tqdm import tqdm
import pickle

class Config:
    """Configuration class - adjust paths as needed"""
    def __init__(self):
        self.data_path = "../00_data/"
        self.esco_path = "../00_data/ESCO/"
        self.models_path = "../00_data/SBERT_Models/models/"

def flatten_list(list_of_lists):
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return flattened_list

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def load_data_pairs():
    """Load training data pairs from existing structure"""
    path_to_data = "../00_data/SBERT_Models/trainingdata/"
    traindata = {}
    for file in tqdm(os.listdir(path_to_data)):
        if "_pairs" in file:
            traindata[file.split(".")[0]] = load_json(path_to_data + file)
    return traindata

def write_pickle(filepath, data):
    with open(filepath, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Pickle saved.")

def encode_jobs(model):
    """Encode ESCO jobs using existing data structure"""
    jobs = load_json("../00_data/ESCO/ESCO_JOBS_ALL.json")
    embedding_dict = {x: {} for x in ["skillsets", "desc", "jobtitle"]}
    jobtitles = [x["jobtitle"] for x in jobs]
    skillsets = [" ".join(x["full_skills"]) for x in jobs]
    descs = [x["jobdescription"] for x in jobs]
    escoids = [x["jobid_esco"] for x in jobs]

    skill_embeddings = model.encode(skillsets, show_progress_bar=True)
    desc_embeddings = model.encode(descs, show_progress_bar=True)
    title_embeddings = model.encode(jobtitles, show_progress_bar=True)

    embedding_dict["skillsets"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": skill_embeddings})
    embedding_dict["desc"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": desc_embeddings})
    embedding_dict["jobtitle"].update({"jobtitle": jobtitles, "esco_id": escoids, "embeddings": title_embeddings})

    return embedding_dict

def write_json(filename, data):
    with open(filename, 'w', encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print("Successfully saved file:", filename)
    return filename

def load_pickle(filepath):
    with open(filepath, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data