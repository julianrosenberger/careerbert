import json
from pypdf import PdfReader
from pathlib import Path
import pickle

def write_unique_json(filename, data):
    """
    Writes a JSON file with a unique filename by appending an incrementing number if the file already exists.

    Args:
        filename (str): The base filename for the JSON file.
        data (dict): The data to be written to the JSON file.

    Returns:
        str: The unique filename used to save the JSON file.
    """
    i = 0
    while True:
        unique_filename = f"{filename}_{i}.json"
        if not Path(unique_filename).is_file():
            break
        i += 1
    with open(unique_filename, 'w', encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print(f"Successfully saved file: {unique_filename}")
    return unique_filename

def write_json(filename, data):
    """
    Writes data to a JSON file.

    Args:
        filename (str): The filename for the JSON file.
        data (dict): The data to be written to the JSON file.

    Returns:
        str: The filename used to save the JSON file.
    """
    with open(filename, 'w', encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print(f"Successfully saved file: {filename}")
    return filename

def load_json(filename):
    """
    Loads data from a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(filename, "r", encoding="utf-8") as fp:
        return json.load(fp)

def load_pdf(filepath):
    """
    Extracts and concatenates text from all pages of a PDF file.

    Args:
        filepath (str): The path to the PDF file.

    Returns:
        str: The concatenated text from the PDF file.
    """
    reader = PdfReader(filepath)
    return "".join(page.extract_text().strip() for page in reader.pages)

def load_cvs():
    """
    Loads and extracts text from a predefined set of CV PDF files.

    Returns:
        list: A list of strings, each containing the text from a CV.
    """
    cvs = []
    for i in range(1, 6):
        reader = PdfReader(f"00_data/CVs/CV_{i}.pdf")
        cv_text = "".join(page.extract_text().strip() for page in reader.pages)
        cvs.append(cv_text)
    return cvs


def load_jobs():
    """
    Loads job data from a predefined JSON file.

    Returns:
        dict: The job data loaded from the JSON file.
    """
    return load_json("00_data/ESCO/ESCO_JOBS_ALL.json")

def write_pickle(filepath, data):
    """
    Writes data to a pickle file.

    Args:
        filepath (str): The path to the pickle file.
        data (any): The data to be serialized and written to the file.
    """
    with open(filepath, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Pickle saved.")

def load_pickle(filepath):
    """
    Loads data from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        any: The data loaded from the pickle file.
    """
    with open(filepath, "rb") as fIn:
        return pickle.load(fIn)

def flatten_list(list_of_lists):
    """
    Flattens a list of lists into a single list.

    Args:
        list_of_lists (list): A list containing sublists.

    Returns:
        list: A flattened list containing all elements from the sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]

def sort_dict(dictionary, reverse=True):
    """
    Sorts a dictionary by its values.

    Args:
        dictionary (dict): The dictionary to be sorted.
        reverse (bool): Whether to sort in descending order. Defaults to True.

    Returns:
        dict: A new dictionary sorted by values.
    """
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=reverse)}

def esco_lookup(item):
    """
    Looks up an item in the ESCO lookup dictionary.

    Args:
        item (str): The item to look up.

    Returns:
        str or None: The corresponding ESCO ID if found, otherwise None.
    """
    esco_lookup_dict = load_json("00_data/ESCO/esco_lookup.json")
    return esco_lookup_dict.get(item.lower(), None)

def add_esco_jobids(ad):
    """
    Adds ESCO job IDs to a given job or list of jobs.

    Args:
        ad (str or list): A job title (str) or a list of job titles.

    Returns:
        str or list or None: The ESCO job ID(s) if found, otherwise None.
    """
    global esco_lookup_dict
    if "esco_lookup_dict" not in globals():
        esco_lookup_dict = load_json("00_data/ESCO/esco_lookup.json")
    
    if isinstance(ad, str):
        return esco_lookup_dict.get(ad, None)
    
    esco_ids = [esco_lookup_dict[job] for job in ad if job in esco_lookup_dict and "." in esco_lookup_dict[job]]
    return esco_ids if esco_ids else None
