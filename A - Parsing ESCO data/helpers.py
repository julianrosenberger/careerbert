import requests
import json
from pathlib import Path
from tqdm import tqdm

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
        json.dump(data, fp,indent=2, ensure_ascii=False)
    print("Sucessfully saved file:",checkfilename)
    return checkfilename


def load_json(filename):
    with open(filename,"r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data



def get_all_jobs():

    visitedlevels = []
    url = "https://ec.europa.eu/esco/api/resource/taxonomy?uri=http://data.europa.eu/esco/concept-scheme/occupations&language=de&selectedVersion=v1.1.0"
    r = requests.get(url)
    data = r.json()
    topleveluris = []

    for item in data["_links"]["hasTopConcept"]:
        topleveluris.append({"title":item["title"], "uri": item["uri"]})

    for TLuri in topleveluris:
        jobs =  []
        if Path("jsons/"+TLuri["title"]+"_0.json").is_file():
            print("Skipped",TLuri["title"])
            continue
        queue = [TLuri["uri"]]
        while queue:
            uri = queue.pop(0)
            visitedlevels.append(uri)
            r = requests.get(f"https://ec.europa.eu/esco/api/resource/occupation?uri={uri}&language=de&selectedVersion=v1.1.0")
            data = r.json()["_links"]
            if "narrowerConcept" in data:
                for item in data["narrowerConcept"]:
                    queue.append(item["uri"])
            if "narrowerOccupation" in data:
                for item in data["narrowerOccupation"]:
                    jobs.append({"jobtitle":item["title"],"uri":item["uri"]})
        write_unique_json(TLuri["title"], jobs)
    return visitedlevels

def get_jobdata(berufe,language):
    alljobinfo = []
    errors = []
    for job in tqdm(berufe):
        try:
            uri = job["uri"]
            r = requests.get("https://ec.europa.eu/esco/api/resource/occupation?uri="+uri+"&language="+language+"&selectedVersion=v1.1.0")
            data = r.json()

            jobtitle = data["title"]
            jobdescription = data["description"][language]["literal"]

            essential_skills = []
            [essential_skills.append(item["title"]) for item  in data["_links"]["hasEssentialSkill"]]
            additional_skills = []
            [additional_skills.append(item["title"]) for item  in data["_links"]["hasOptionalSkill"]]

            id = data["code"]
            jobcategory = data["_links"]["broaderIscoGroup"][0]["title"]
            jobcategoryid = data["_links"]["broaderIscoGroup"][0]["code"]

            jobinfo = {"jobid_esco": id,"jobtitle":jobtitle, "skills":essential_skills,"additional_skills":additional_skills,
            "jobdescription":jobdescription,"jobcategory": jobcategory, "jobcategoryid":jobcategoryid}

            #print(jobinfo)
            alljobinfo.append(jobinfo)
        except KeyboardInterrupt:
            print("Interrupted")
            write_unique_json("parsedjobinfo_"+language, alljobinfo)
            write_unique_json("errors_"+language, errors)
            return alljobinfo, errors
        except:
            print("Error", uri)
            errors.append({"uri": uri})
    write_unique_json("parsedjobinfo_"+language, alljobinfo)
    write_unique_json("errors_" + language, errors)
    return alljobinfo, errors
