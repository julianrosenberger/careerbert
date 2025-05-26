import requests
from pathlib import Path
from helpers import write_unique_json

def get_all_jobs():

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