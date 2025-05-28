import requests
from pathlib import Path
from tqdm import tqdm
from .._utils import write_unique_json

def get_jobdata(berufe, language):
    """
    Fetches and processes job data from the ESCO API for a list of occupations.
    This function retrieves detailed information about each job in the provided list,
    including job title, description, essential and optional skills, job category,
    and related identifiers. It processes the data into a structured format and
    saves the results to JSON files.
    Parameters:
    -----------
    berufe : list
        A list of dictionaries, each containing at least a 'uri' key that points
        to the ESCO API resource for a specific occupation.
    language : str
        The language code (e.g., 'en', 'de', 'fr') to retrieve the job information in.
    Returns:
    --------
    tuple
        A tuple containing two elements:
        - alljobinfo: A list of dictionaries with processed job information.
        - errors: A list of dictionaries containing URIs that caused errors during processing.
    Notes:
    ------
    - The function shows a progress bar using tqdm.
    - Results are automatically saved to JSON files with names like "parsedjobinfo_{language}.json"
        and "errors_{language}.json".
    - The function handles keyboard interrupts by saving partial results before exiting.
    - API data is retrieved from the ESCO API v1.1.0.
    """
    occupation_data_list = []
    failed_requests = []
    for occupation in tqdm(berufe):
        try:
            occupation_uri = occupation["uri"]
            response = requests.get("https://ec.europa.eu/esco/api/resource/occupation?uri="+occupation_uri+"&language="+language+"&selectedVersion=v1.1.0")
            occupation_details = response.json()

            occupation_title = occupation_details["title"]
            occupation_description = occupation_details["description"][language]["literal"]

            essential_skills = []
            [essential_skills.append(skill["title"]) for skill in occupation_details["_links"]["hasEssentialSkill"]]
            optional_skills = []
            [optional_skills.append(skill["title"]) for skill in occupation_details["_links"]["hasOptionalSkill"]]

            occupation_id = occupation_details["code"]
            occupation_category = occupation_details["_links"]["broaderIscoGroup"][0]["title"]
            occupation_category_id = occupation_details["_links"]["broaderIscoGroup"][0]["code"]

            occupation_info = {
                "jobid_esco": occupation_id,
                "jobtitle": occupation_title, 
                "skills": essential_skills,
                "additional_skills": optional_skills,
                "jobdescription": occupation_description,
                "jobcategory": occupation_category, 
                "jobcategoryid": occupation_category_id
            }

            occupation_data_list.append(occupation_info)
        except KeyboardInterrupt:
            print("Interrupted")
            write_unique_json("parsedjobinfo_"+language, occupation_data_list)
            write_unique_json("errors_"+language, failed_requests)
            return occupation_data_list, failed_requests
        except:
            print("Error", occupation_uri)
            failed_requests.append({"uri": occupation_uri})
    write_unique_json("parsedjobinfo_"+language, occupation_data_list)
    write_unique_json("errors_" + language, failed_requests)
    return occupation_data_list, failed_requests
