import os
from tqdm import tqdm
import sys
sys.path.append("..")
from _utils import load_json

def load_data_pairs():
    """
    Loads training data pairs from JSON files in the specified directory.

    The function scans the directory "../00_data/SBERT_Models/trainingdata/" for files
    containing "_pairs" in their names. It loads the content of these files as JSON
    and stores them in a dictionary where the keys are the filenames (without extensions)
    and the values are the loaded JSON data.

    Returns:
        dict: A dictionary containing the training data pairs.
    """
    path_to_data = "../00_data/SBERT_Models/trainingdata/"
    traindata = {}
    for file in tqdm(os.listdir(path_to_data)):
        if "_pairs" in file:
            traindata[file.split(".")[0]] = load_json(path_to_data + file)
    return traindata

def create_trainig_samples(pos_dev_samples, neg_pairs):
    """
    Creates a development set for training by combining positive and negative pairs.

    The function filters positive and negative pairs based on shared anchors (queries).
    It then creates a list of dictionaries where each dictionary contains a query,
    its associated positive samples, and its associated negative samples.

    Args:
        pos_dev_samples (list): A list of positive development samples, where each sample
                                is a tuple (anchor, positive_sample).
        neg_pairs (list): A list of negative pairs, where each pair is a tuple (anchor, negative_sample).

    Returns:
        list: A list of dictionaries, each containing a query, positive samples, and negative samples.
    """
    dev_set_total = []
    anchors = set([x[0] for x in pos_dev_samples])
    neg_dev_samples = [x for x in neg_pairs if x[0] in anchors]
    print("Creating Devset")
    for anchor in tqdm(anchors):
        pos_pairs_filtered = [x[1] for x in pos_dev_samples if x[0] == anchor]
        neg_pairs_filtered = [x[1] for x in neg_dev_samples if x[0] == anchor]
        dev_set_total.append({"query": anchor, "positive": pos_pairs_filtered, "negative": neg_pairs_filtered})
    return dev_set_total

def encode_jobs(model):
    """
    Encodes job-related data into embeddings using a given model.

    The function loads job data from a JSON file and extracts job titles, skill sets,
    descriptions, and ESCO IDs. It then uses the provided model to encode these fields
    into embeddings. The embeddings are stored in a dictionary categorized by "skillsets",
    "desc", and "jobtitle".

    Args:
        model: A model object with an `encode` method that generates embeddings for input text.

    Returns:
        dict: A dictionary containing embeddings for skill sets, descriptions, and job titles,
              along with their corresponding job titles and ESCO IDs.
    """
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