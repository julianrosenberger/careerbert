import requests
from pathlib import Path
from helpers import write_unique_json

def get_all_jobs():
    """
    Retrieves and processes job occupations from the ESCO API.
    This function fetches all job occupations from the European Skills, Competences, Qualifications and Occupations (ESCO) API.
    It first retrieves the top 10 levels of the ESCO occupation hierarchy, then iteratively traverses down the hierarchy
    to collect all job titles and their URIs. The results are saved in JSON files, with one file for each top level category.
    If a JSON file for a top level already exists, that top level is skipped to avoid redundant processing.
    Returns:
        None: Results are written directly to JSON files.
    Note:
        - Requires the 'requests' library for API calls
        - Requires the 'Path' class from the 'pathlib' module
        - Depends on an external 'write_unique_json' function to save results
    """

    url = "https://ec.europa.eu/esco/api/resource/taxonomy?uri=http://data.europa.eu/esco/concept-scheme/occupations&language=de&selectedVersion=v1.1.0"
    r = requests.get(url, timeout=20)
    data = r.json()
    topleveluris = []

    # fetch the 10 top levels of ESCO (0 - 9)
    for item in data["_links"]["hasTopConcept"]:
        topleveluris.append({"title":item["title"], "uri": item["uri"]})

    # for every top level, create a queue for the deeper levels
    # safe all jobs encountered in jobs
    for TLuri in topleveluris:
        jobs =  []
        #if json file for top level already exists, skip to the next.
        if Path(TLuri["title"]+"_0.json").is_file():
            print("Skipped",TLuri["title"])
            continue
        queue = [TLuri["uri"]]
        while queue:
            print(len(queue), TLuri["title"])
            uri = queue.pop(0)
            #visitedlevels.append(uri)
            r = requests.get(f"https://ec.europa.eu/esco/api/resource/occupation?uri={uri}&language=de&selectedVersion=v1.1.0")
            data = r.json()["_links"]
            if "narrowerConcept" in data:
                for item in data["narrowerConcept"]:
                    queue.append(item["uri"])
            if "narrowerOccupation" in data:
                for item in data["narrowerOccupation"]:
                    jobs.append({"jobtitle":item["title"],"uri":item["uri"]})
                    queue.append(item["uri"])
        write_unique_json(TLuri["title"], jobs)
    #return visitedlevels

if __name__ == "__main__":
    get_all_jobs()