from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from tqdm import tqdm
import re
import requests
from _util import *
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import math

def get_joburls_ams(jobtitles):
    ids = []
    driver = webdriver.Chrome(ChromeDriverManager().install())
    urls = []
    for job in tqdm(jobtitles):
        try:
            driver.get(f"https://jobs.ams.at/public/emps/jobs?query={job}&JOB_OFFER_TYPE=SB_WKO&JOB_OFFER_TYPE=IJ&JOB_OFFER_TYPE=BA")
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source)
            results = soup.find(id="ams-search-result-text")
            result_no = results.text.split(" ")[1]
            pagenumber = math.ceil((int(result_no))/10)
            print(f"Found {result_no} results for {job}")
            total = 0
            for p_no in range(1,pagenumber+1):
                driver.get(f"https://jobs.ams.at/public/emps/jobs?page={p_no}&query={job}&JOB_OFFER_TYPE=SB_WKO&JOB_OFFER_TYPE=IJ&JOB_OFFER_TYPE=BA&PERIOD=ALL&sortField=_SCORE")
                time.sleep(1)
                soup = BeautifulSoup(driver.page_source)
                jobhits = soup.find_all(id = re.compile(r"ams-search-joboffer-.?-title-link"))
                berufsbezeichnung = soup.find_all(id = "ams-search-joboffer-occupation")
                urls_page = [{"jobtitle":job,"title_parsed": jobhit.text,"ams_job_id":jobhit["href"].split("/")[-1], "url": "https://jobs.ams.at"+jobhit["href"],"beruf":occupation.text} for jobhit,occupation  in zip(jobhits,berufsbezeichnung) if jobhit["href"].split("/")[-1] not in ids]
                ids_page = [jobhit["href"].split("/")[-1] for jobhit in jobhits]
                ids +=ids_page
                total += len(urls_page)
                #print(f"Found {total} results for {job}")
                urls +=urls_page
                if total >= 200:
                    break
        except KeyboardInterrupt:
            driver.quit()
            return urls
        except:
            continue
    driver.quit()
    return urls

def driver_setup(headless=True):
    
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    if headless != False:
        options.add_argument('--headless')
    
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)

    return driver

def parse_multithreading_ams(urllist):
    drivers = [driver_setup() for _ in range(4)]
    chunks = np.array_split(urllist, 4)
    with ThreadPoolExecutor(max_workers=4) as executor:  
        bucket = executor.map(add_jobdescriptions, chunks, drivers)
        results = [item for block in bucket for item in block]
    
    [driver.quit() for driver in drivers]
    return results

def add_jobdescriptions(urllist, driver):
    #driver = webdriver.Chrome(ChromeDriverManager().install())
    for item in tqdm(urllist):
        try:
            driver.get(item["url"])
            time.sleep(0.5)
            soup = BeautifulSoup(driver.page_source)
            description = soup.find(id ="ams-detail-jobdescription-text").text
        except:
            description = None
        item.update({"description":description})
    driver.quit()
    return urllist

def get_all_esco_levels():
    url = "https://ec.europa.eu/esco/api/resource/taxonomy?uri=http://data.europa.eu/esco/concept-scheme/occupations&language=de&selectedVersion=v1.1.0"
    r = requests.get(url)
    data = r.json()
    TLURIS = []
    for item in data["_links"]["hasTopConcept"]:
        TLURIS.append([item["title"],item["uri"]])
    alldata = []
    for TLU in tqdm(TLURIS):
        TLU_DATA = []
        queue = [TLU[1]]
        while queue:
            print(len(queue))
            uri = queue.pop(0)
            r = requests.get(f"https://ec.europa.eu/esco/api/resource/occupation?uri={uri}&language=de&selectedVersion=v1.1.0")
            time.sleep(0.5)
            data = r.json()["_links"]
            if "narrowerConcept" in data:
                for item in data["narrowerConcept"]:
                    queue.append(item["uri"])
                    TLU_DATA.append({"levelname":item["title"],"code": item["code"]})
        alldata += TLU_DATA
        write_unique_json(f"{TLU[0]}_escolevels", TLU_DATA)
    newlist = sorted(alldata, key=lambda d: d['code'])
    return newlist

def parse_multithreading_eres(urllist):
    drivers = [driver_setup() for _ in range(4)]
    chunks = np.array_split(urllist, 4)
    with ThreadPoolExecutor(max_workers=4) as executor:  
        bucket = executor.map(get_jobdescription_eres, chunks, drivers)
        results = [item for block in bucket for item in block]
    
    [driver.quit() for driver in drivers]
    return results

def get_jobdescription_eres(ads, driver):
    results = []
    for ad in tqdm(ads):
        driver.get(ad["url"])
        time.sleep(8)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        try:
            jobtitle = soup.find("h1").text
        except:
            jobtitle = None
        try:
            description = soup.find(id="jv-details-job-description").text
        except:
            description = None
        try:
            jobs = soup.find_all(id="jv-job-categories-codes")
            ESCO_JOBS = [job.text.strip("Beruf: ") for job in jobs]
        except:
            ESCO_JOBS = None
        results.append({"ESCO_ID":ad["esco_id"], "parsed_title":jobtitle, "ESCOJOB":ESCO_JOBS, "description": description})
    return results



# def get_linkedin_profile_text(link):
#     print("Fetching started.")
#     driver = driver_setup(headless=True)
#     if "linkedin" not in link:
#         raise TypeError("Link does not seem to be a linkedin profile link.")
#     try:
#         driver.get(link)
#         WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="public_profile_contextual-sign-in"]/div/section/button')))
#         driver.find_element_by_xpath('//*[@id="public_profile_contextual-sign-in"]/div/section/button').click()
#         soup = BeautifulSoup(driver.page_source,features="lxml")
#         driver.quit()
#         print("Fetching completed.")
#     except:
#         print("Error while fetching LinkedIn Profile")
#         return
#     ## SUMMARY
#     try:
#         summary = soup.find(attrs={"data-section":"summary"}).text.replace("\n"," ").replace("  ","")
#     except AttributeError:
#         summary = ""

#     ## POSITIONS
#     sections = ["currentPositionsDetails","pastPositionsDetails"]
#     positions = []
#     for sec in sections:
#         positions += soup.find_all("li", attrs={"data-section":sec})
#     position_text = ""
#     for position in positions:
#         jobname = position.find("h3").text.replace("\n"," ").replace("  ","")
#         try:
#             position_details = position.find("div", {"class":"experience-item__description experience-item__meta-item"}).text.replace("\n"," ").replace("  ","")
#         except AttributeError:
#             position_details = ""
#         position_text += " ".join([jobname, position_details])  
    
#     # Education
#     try:
#         educations = soup.find("ul",{"class":"education__list"}).find_all("li", {"class": "profile-section-card education__list-item"})
#         full_text_educations = ""
#         for ed in educations:
#             ed_spans = ed.find("h4").find_all("span")
#             full_text_educations = ""
#             for span in ed_spans:
#                 full_text_educations += span.text.replace("\n"," ").replace("  ","")
#                 full_text_educations += " "
#             try:
#                 description = ed.find("div", {"class":"education__item--details"}).text.replace("\n"," ").replace("  ","")
#             except:
#                 description = " "
#         full_text_educations += description
#     except:
#         full_text_educations=""
    
#     total_text = summary + position_text + full_text_educations
#     return total_text
        