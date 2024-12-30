import random
import numpy as np
import json
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import os
import pandas as pd

# File loading functions
def load_job_ads(file_path):
    """Load job ads from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_isco_groups(file_path):
    """Load ISCO groups from CSV file"""
    df = pd.read_csv(file_path, dtype={"code": str})
    return create_isco_hierarchy(df)

def create_isco_hierarchy(df):
    """Create hierarchical structure from ISCO groups"""
    hierarchy = {}
    
    # Create 4-level hierarchy based on code length
    for _, row in df.iterrows():
        code = str(row['code'])
        level = len(str(row['code']))
        
        if level == 1:  # Major group
            hierarchy[code] = {
                'label': row['preferredLabel'],
                'subgroups': {},
                'description': row.get('description', '')
            }
        elif level == 2:  # Sub-major group
            major = code[0].zfill(4)
            if major in hierarchy:
                hierarchy[major]['subgroups'][code] = {
                    'label': row['preferredLabel'],
                    'subgroups': {},
                    'description': row.get('description', '')
                }
    return hierarchy

def prepare_test_job_ads(job_ads, n_samples=5, seed=42):
    """Select random job ads with fixed seed"""
    random.seed(seed)
    np.random.seed(seed)
    
    test_ads = random.sample(job_ads, n_samples)
    
    formatted_ads = []
    for ad in test_ads:
        formatted_ad = {
            'job_title': ad['title'],
            'esco_id': ad['esco_id'],  
            'description': ad['short_texts'],
            'correct_esco_job': ad['esco_job']
        }
        formatted_ads.append(formatted_ad)
    
    return formatted_ads

# Prompt creation functions
def create_major_group_prompt(job_ad, hierarchy):
    """Create prompt for identifying major group"""
    prompt = f"""As a job classification expert, identify the most relevant ISCO major groups for this job posting.

Job Title: {job_ad['job_title']}
Job Description: {job_ad['description']}

Available Major Groups:
{json.dumps({code: data['label'] for code, data in hierarchy.items()}, indent=2, ensure_ascii=False)}

Return the TWO most relevant major group codes with confidence scores as JSON:
[
    {{"code": "2", "confidence": 0.9}},
    {{"code": "3", "confidence": 0.7}}
]"""
    return prompt

def create_subgroup_prompt(job_ad, major_code, hierarchy):
    """Create prompt for identifying subgroups within major group"""
    subgroups = hierarchy[major_code]['subgroups']
    
    prompt = f"""Within the major group "{hierarchy[major_code]['label']}", identify the most relevant sub-major groups for this job.

Job Title: {job_ad['job_title']}
Job Description: {job_ad['description']}

Available Sub-major Groups:
{json.dumps({code: data['label'] for code, data in subgroups.items()}, indent=2, ensure_ascii=False)}

Return the most relevant sub-group codes with confidence scores as JSON:
[
    {{"code": "21", "confidence": 0.95}},
    {{"code": "23", "confidence": 0.6}}
]"""
    return prompt

def create_final_matching_prompt(job_ad, esco_jobs):
    """Create prompt for final ESCO job matching"""
    prompt = f"""Match this job posting to the most relevant ESCO jobs.

Job Title: {job_ad['job_title']}
Job Description: {job_ad['description']}

Available ESCO Jobs:
{json.dumps(esco_jobs, indent=2, ensure_ascii=False)}

Return the top 100 matches as JSON:
[
    {{"esco_id": "string", "confidence": float}},
    ...
]"""
    return prompt

async def process_single_job(job_ad, hierarchy, esco_jobs, client):
    """Process a single job ad through the hierarchical matching"""
    try:
        # Step 1: Identify major groups
        major_prompt = create_major_group_prompt(job_ad, hierarchy)
        major_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": major_prompt}],
            temperature=0
        )
        major_groups = json.loads(major_response.choices[0].message.content)
        
        relevant_esco_jobs = []
        
        # Step 2: For each major group, identify relevant subgroups
        for major in major_groups:
            major_code = major['code'].zfill(4)
            subgroup_prompt = create_subgroup_prompt(job_ad, major_code, hierarchy)
            
            subgroup_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": subgroup_prompt}],
                temperature=0
            )
            subgroups = json.loads(subgroup_response.choices[0].message.content)
            
            # Collect ESCO jobs from relevant subgroups
            for subgroup in subgroups:
                relevant_jobs = [
                    job for job in esco_jobs 
                    if job['jobid_esco'].startswith(subgroup['code'])
                ]
                relevant_esco_jobs.extend(relevant_jobs)
        
        # Step 3: Final matching
        final_prompt = create_final_matching_prompt(job_ad, relevant_esco_jobs)
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0
        )
        
        matches = json.loads(final_response.choices[0].message.content)
        
        # Calculate rank and MRR
        correct_id = job_ad['esco_id']
        rank = None
        for idx, match in enumerate(matches, 1):
            if match['esco_id'] == correct_id:
                rank = idx
                break
        
        mrr = 1/rank if rank and rank <= 100 else 0
        
        return {
            'job_title': job_ad['job_title'],
            'correct_esco_id': correct_id,
            'predicted_rank': rank,
            'mrr': mrr
        }
        
    except Exception as e:
        print(f"Error processing job {job_ad['job_title']}: {str(e)}")
        return None

async def run_experiment(test_ads, hierarchy, esco_jobs):
    """Run the complete experiment"""
    client = OpenAI()
    results = []
    
    for ad in tqdm(test_ads):
        result = await process_single_job(ad, hierarchy, esco_jobs, client)
        if result:
            results.append(result)
        time.sleep(1)  # Rate limiting
    
    return results

# Main execution
if __name__ == "__main__":
    # Load data
    job_ads = load_job_ads('../00_data/EURES/eures_testads_final_short.json')
    isco_hierarchy = load_isco_groups('../00_data/ESCO/ESCO_isco_groups.csv')
    
    # Prepare test ads
    test_ads = prepare_test_job_ads(job_ads, n_samples=5, seed=42)
    
    # Run experiment
    results = run_experiment(test_ads, isco_hierarchy, job_ads)
    
    # Calculate and display results
    avg_mrr = sum(r['mrr'] for r in results) / len(results)
    print(f"\nAverage MRR@100: {avg_mrr:.3f}")
    
    print("\nDetailed Results:")
    for r in results:
        print(f"\nJob: {r['job_title']}")
        print(f"Correct ESCO ID: {r['correct_esco_id']}")
        print(f"Rank: {r['predicted_rank']}")
        print(f"MRR: {r['mrr']:.3f}")