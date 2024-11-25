import json
from pypdf import PdfReader
from pathlib import Path
import pickle


def write_unique_json(filename,data):
    unique = False
    i = 0
    while unique == False:
        checkfilename = filename+"_"+str(i)+".json"
        if Path(checkfilename).is_file():
            i += 1
        else:
            unique = True
    with open(checkfilename, 'w',encoding= "utf-8") as fp:
        json.dump(data, fp, indent = 2, ensure_ascii=False)
    print("Sucessfully saved file:",checkfilename)
    return checkfilename


def write_json(filename,data):
    with open(filename, 'w',encoding= "utf-8") as fp:
        json.dump(data, fp, indent = 2, ensure_ascii=False)
    print("Sucessfully saved file:",filename)
    return filename

def load_json(filename):
    with open(filename,"r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def load_pdf(filepath):
    reader = PdfReader(filepath)
    # printing number of pages in pdf file 
    # getting a specific page from the pdf file
    pages = reader.pages

    cv=""
    for i in range(len(pages)):
        page = reader.pages[i].extract_text().strip()
        cv +=page
    return cv

def load_cvs():
    cvs = []
    for i in range(1,6):
        cv = ""
        reader = PdfReader(f"../00_data/CVs/CV_{i}.pdf")

        pages = reader.pages
        for i in range(len(pages)):
            page = reader.pages[i].extract_text().strip()
            cv +=page
        cvs.append(cv)
    return cvs

def load_jobs():
    with open(r"..\00_data\ESCO\ESCO_JOBS_ALL.json","r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def write_pickle(filepath, data):
    with open(filepath, "wb") as fOut:
        pickle.dump(data, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Pickle saved.")

def load_pickle(filepath):
    with open(filepath, "rb") as fIn:
        stored_data = pickle.load(fIn)
        return stored_data

def flatten_list(list_of_lists):
    flattened_list =  [item for sublist in list_of_lists for item in sublist]
    return flattened_list

def sort_dict(dict,reverse=True):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=reverse)}

def esco_lookup(item):
    esco_lookup_dict = load_json(r"../00_data/ESCO/esco_lookup.json")
    try:
        return esco_lookup_dict[item.lower()]
    except:
        print("Not found.")
        return None
    
def add_esco_jobids(ad):
    if "esco_lookup_dict" not in globals():
        global esco_lookup_dict
        esco_lookup_dict = load_json(r"../00_data/ESCO/esco_lookup.json")
    escoid = []
    if isinstance(ad,str):
        try:
            id = esco_lookup_dict[ad]
            return id
        except:
            return None
    for job in ad:
        try:
            id = esco_lookup_dict[job]
            if "." in id:
                escoid.append(id)
        except:
            continue
    if escoid:
        return escoid
    else:
        return None
