from bs4 import BeautifulSoup
import time
# from webdriver_manager.chrome import ChromeDriverManager
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
from selenium.webdriver import Chrome

def driver_setup(headless=True):
    
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    if headless != False:
        options.add_argument('--headless')
    
    driver = Chrome(options=options)

    return driver

def get_jobdescription_eures(ads, driver):
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