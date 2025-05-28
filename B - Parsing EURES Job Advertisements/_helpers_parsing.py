from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from selenium.webdriver import Chrome, ChromeOptions
import sys
sys.path.append('..')
from _utils import write_json, load_json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import numpy as np

def driver_setup(headless=True):
    """
    Set up and configure a Chrome WebDriver instance.
    
    Args:
        headless (bool): Whether to run Chrome in headless mode. Default is True.
        
    Returns:
        Chrome: Configured Chrome WebDriver instance.
    """
    options = ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    if headless:
        options.add_argument('--headless')
    
    driver = Chrome(options=options)
    return driver


def extend_jobs(ad):
    esco_lookup_dict = load_json(r"../00_data/ESCO/esco_lookup.json")
    extended = []
    for job in ad["esco_jobs"]:
        job_ext = dict(ad)
        job_ext["esco_job"] = job
        try:
            id = esco_lookup_dict[job.lower()]
            job_ext["esco_id"] = id
        except:
            job_ext["esco_id"] = None
            continue
        if "." in id:
            extended.append(job_ext)
    return extended

# Function to parse job advertisements using multithreading
def parse_multithreading_eures(job_ads_df, already_parsed_urls, headless=True):
    """
    Parse job advertisements using multithreading and Selenium web drivers.

    Args:
        job_ads_df (pd.DataFrame): DataFrame containing job advertisements with URLs and metadata.
        already_parsed_urls (list): List of URLs that have already been parsed.
        headless (bool): Whether to run the Selenium WebDriver in headless mode. Default is True.

    Returns:
        list: List of parsed job advertisements with additional details.
    """
    # Initialize Selenium drivers
    drivers = [driver_setup(headless) for _ in range(4)]

    # Generate unique filenames for intermediate results
    timestamp = "".join([char for char in str(datetime.now()).split('.')[0] if char.isdigit()])
    output_files = [f"../00_data/EURES/{timestamp}_parsed_ads_{i}.json" for i in range(1, 5)]

    print("Filtering out already parsed job advertisements.")
    unparsed_job_ads = job_ads_df[~job_ads_df["url"].isin(already_parsed_urls)].to_dict("records")
    print(f"Number of job advertisements to parse: {len(unparsed_job_ads)}")

    # Split the job advertisements into chunks for multithreading
    job_ad_chunks = np.array_split(unparsed_job_ads, 4)

    # Use ThreadPoolExecutor to parse job advertisements concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        parsed_chunks = executor.map(fetch_job_descriptions, job_ad_chunks, drivers, output_files)
        parsed_results = [job_ad for chunk in parsed_chunks for job_ad in chunk]

    # Save the combined results to a single JSON file
    output_path = f"../00_data/EURES/{timestamp}_parsed_ads_total.json"
    write_json(output_path, parsed_results)

    return parsed_results

# Function to fetch job descriptions and related details
def fetch_job_descriptions(job_ads, web_driver, output_file):
    """
    Fetch job descriptions and related details from EURES job advertisements.

    Args:
        job_ads (list): List of job advertisement dictionaries containing URLs and other metadata.
        web_driver (webdriver): Selenium WebDriver instance for web scraping.
        output_file (str): Path to the output file where parsed results will be saved.

    Returns:
        list: List of parsed job advertisements with additional details.
    """
    parsed_results = []
    dead_links = []

    for job_ad in tqdm(job_ads):
        web_driver.get(job_ad["url"] + "?jvDisplayLanguage=de&lang=de")

        # Check for dead links
        try:
            WebDriverWait(web_driver, 3).until(
                EC.presence_of_element_located((By.ID, "error-message-jv-detail"))
            )
            dead_links.append(job_ad)
            write_json(output_file + "_deadlinks", dead_links)
            continue
        except:
            pass

        # Wait for job description and ESCO job categories to load
        try:
            WebDriverWait(web_driver, 8).until(
                EC.presence_of_element_located((By.ID, "jv-details-job-description"))
            )
            WebDriverWait(web_driver, 8).until(
                EC.presence_of_element_located((By.ID, "jv-job-categories-codes"))
            )
            time.sleep(0.5)
            page_content = BeautifulSoup(web_driver.page_source, "html.parser")
        except:
            continue

        # Extract job title
        job_title = page_content.find("h1").text if page_content.find("h1") else None

        # Extract job description
        job_description = None
        try:
            description_content = page_content.find(id="jv-details-job-description").contents
            job_description = "".join(
                content.text.replace("\xa0", "") if hasattr(content, 'text') else ""
                for content in description_content
            )
        except:
            pass

        # Extract ESCO job categories
        esco_jobs = None
        try:
            esco_jobs_container = page_content.find(id="jv-job-categories-codes")
            esco_jobs_list = esco_jobs_container.find_all(class_="ecl-u-ml-2xs ng-star-inserted")
            esco_jobs = [job.text.replace("  -", "").strip() for job in esco_jobs_list]
        except:
            pass

        # Update job advertisement with parsed details
        job_ad.update({
            "parsed_title": job_title,
            "esco_jobs": esco_jobs,
            "description": job_description
        })
        parsed_results.append(job_ad)

        # Save intermediate results every 50 records
        if len(parsed_results) % 50 == 0:
            write_json(output_file, parsed_results)

    # Save final results
    write_json(output_file, parsed_results)
    web_driver.quit()
    return parsed_results