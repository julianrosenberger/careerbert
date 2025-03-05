from helpers import *
import os
import random
random.seed(10)
import pandas as pd
import numpy as np
import json
from tqdm import tqdm 
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

load_dotenv()

# Config
client = OpenAI()
MODEL = "gpt-4o-mini"
MRR_AT = 100

# Load test ads
testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))

# Load ESCO jobs (adjust path as needed)
with open("../00_data/ESCO/ESCO_JOBS_ALL.json", 'r') as f:
    esco_jobs = json.load(f)

def create_batch_prompt(job_ad: str, batch: List[Dict], batch_num: int, total_batches: int) -> str:
    """Create prompt for batch processing"""
    return """Analysieren Sie diese zusammengefasste EURES Stellenbeschreibung und identifizieren Sie passende ESCO-Berufe aus dem bereitgestellten Batch.
    Geben Sie nur Matches zurück, die wirklich relevant erscheinen.

    Stellenbeschreibung:
    {}

    ESCO-Berufe Batch {}/{}:
    {}

    Geben Sie die Matches als JSON-Objekt in diesem Format zurück:
    {{
        "matches": [
            {{
                "esco_id": "id",
                "confidence": bewertung_0_bis_100,
                "reasoning": "kurze_begruendung_auf_deutsch"
            }}
        ]
    }}

    Antworten Sie ausschließlich mit einem JSON-Objekt.""".format(
        job_ad,
        batch_num,
        total_batches,
        json.dumps([{
            'id': job['jobid_esco'],
            'title': job['jobtitle'],
            'description': job['jobdescription']
        } for job in batch], ensure_ascii=False)
    )

def create_ranking_prompt(job_ad: str, matches: List[Dict]) -> str:
    """Create prompt for final ranking"""
    return """Erstellen Sie ein finales Ranking der relevantesten Jobmatches.

    Stellenbeschreibung:
    {}

    Potentielle Matches:
    {}

    Bitte geben Sie genau die 100 besten Matches als JSON-Objekt zurück, sortiert nach Relevanz.
    Format:
    {{
        "matches": [
            {{
                "esco_id": "id",
                "confidence": bewertung_0_bis_100,
                "reasoning": "kurze_begruendung"
            }}
        ]
    }}
    
    Antworten Sie ausschließlich mit einem JSON-Objekt.""".format(job_ad, json.dumps(matches, ensure_ascii=False))

def match_job_batched(job_ad: str, esco_jobs: List[Dict], batch_size: int = 50, verbose: bool = False) -> List[Dict]:
    """Match jobs by processing ESCO jobs in batches and combining results"""
    all_matches = []
    total_batches = (len(esco_jobs) + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Processing {total_batches} batches of ESCO jobs...")
    
    # Process ESCO jobs in batches
    for i in range(0, len(esco_jobs), batch_size):
        batch = esco_jobs[i:i+batch_size]
        batch_num = i//batch_size + 1
        
        prompt = create_batch_prompt(job_ad, batch, batch_num, total_batches)

        if verbose:
            print(f"\nVerarbeite Batch {batch_num}/{total_batches}")
        if verbose and batch_num == 1:
            print("\nPrompt Preview:")
            print(prompt[:500] + "...")
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Sie sind ein Experte für Jobmatching und analysieren Stellenbeschreibungen in deutscher Sprache. Antworten Sie ausschließlich mit JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            try:
                batch_matches = json.loads(response.choices[0].message.content)['matches']
                if verbose:
                    print(f"Gefunden: {len(batch_matches)} Matches in diesem Batch")
                all_matches.extend(batch_matches)
            except Exception as e:
                print(f"Fehler beim Parsen der Batch-Matches {batch_num}: {e}")
                continue
            
        except Exception as e:
            print(f"Fehler bei Batch {batch_num}: {e}")
            continue
    
    # Final ranking with error handling
    if len(all_matches) > 0:
        if verbose:
            print(f"\nErstelle finales Ranking aus {len(all_matches)} potentiellen Matches...")
        
        try:
            ranking_prompt = create_ranking_prompt(job_ad, all_matches)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Sie sind ein Experte für das Ranking von Jobmatches. Antworten Sie ausschließlich mit JSON."},
                    {"role": "user", "content": ranking_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            try:
                final_matches = json.loads(response.choices[0].message.content)['matches']
                if verbose:
                    print(f"Finales Ranking erstellt mit {len(final_matches)} Matches")
                return final_matches[:100]
            except Exception as e:
                print(f"Fehler beim Parsen des Rankings: {e}")
                all_matches.sort(key=lambda x: x['confidence'], reverse=True)
                return all_matches[:100]
            
        except Exception as e:
            print(f"Fehler beim finalen Ranking: {e}")
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            return all_matches[:100]
    
    return []

def evaluate_match(matches: List[Dict], true_esco_id: str, mrr_at: int = 100) -> Dict:
    """Evaluate matching results against true ESCO ID.
    
    Args:
        matches (List[Dict]): Sorted list of matches with confidence scores
        true_esco_id (str): Correct ESCO ID to evaluate against
        mrr_at (int): MRR calculation cutoff. Defaults to 100.
    
    Returns:
        Dict: Evaluation metrics including rank, MRR and hit rates
    """
    # Use more efficient list comprehension to find rank
    ranks = [i for i, m in enumerate(matches, 1) if m['esco_id'] == true_esco_id]
    correct_rank = ranks[0] if ranks else None
    
    # Calculate metrics
    mrr = 1/correct_rank if correct_rank and correct_rank <= mrr_at else 0
    missing = 1 if not correct_rank or correct_rank > mrr_at else 0
    
    return {
        'correct_rank': correct_rank,
        'mrr': mrr,
        'missing': missing,
        'top_1_hit': correct_rank == 1,
        'top_5_hit': correct_rank is not None and correct_rank <= 5,
        'top_20_hit': correct_rank is not None and correct_rank <= 20,
        'top_100_hit': correct_rank is not None and correct_rank <= 100
    }


def test_single_job():
    """Test the matching system with a single job"""
    
    # Get first job
    first_job = testads.iloc[1111]
    
    print("Testing job matching with first job...")
    matches = match_job_batched(first_job['short_texts'], esco_jobs, batch_size=50, verbose=True)
    
    print("\nMatching Results:")
    for i, match in enumerate(matches[:10], 1):  # Show top 10 matches
        print(f"\n{i}. ESCO ID: {match['esco_id']}")
        print(f"   Confidence: {match['confidence']}")
        print(f"   Reasoning: {match['reasoning']}")
    
    # Evaluate the match
    evaluation = evaluate_match(matches, first_job['esco_id'])
    print("\nEvaluation:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value}")


def test_batched_matching():
    """Test the batched matching system with multiple jobs""" 
    # Process all jobs
    MRR = []
    for idx in range(10):  # Process first 10 jobs
        job = testads.iloc[idx]
        
        print(f"\nProcessing job {idx+1}...")
        matches = match_job_batched(job['short_texts'], esco_jobs, batch_size=50, verbose=True)
        
        if not matches:
            continue
        
        evaluation = evaluate_match(matches, job['esco_id'])
        MRR.append(evaluation['mrr'])
        
        print("\nEvaluation:")
        for metric, value in evaluation.items():
            print(f"{metric}: {value}")
    
    print("\nMean MRR:", np.mean(MRR))

def process_jobs(
    testads: pd.DataFrame, 
    esco_jobs: List[Dict], 
    batch_size: int = 50, 
    number_of_jobs: int = 20
) -> pd.DataFrame:
    """Process test job ads and evaluate matches against ESCO occupations.
    
    Args:
        testads (pd.DataFrame): DataFrame containing test job advertisements
        esco_jobs (List[Dict]): List of ESCO occupation dictionaries
        batch_size (int, optional): Number of ESCO jobs per batch. Defaults to 50.
        number_of_jobs (int, optional): Number of jobs to process. Defaults to 20.
        
    Returns:
        pd.DataFrame: Evaluation metrics including MRR, ranks and hit rates
    """
    MRR = []
    results_df = pd.DataFrame()
    
    for idx in tqdm(random.sample(range(len(testads)), number_of_jobs), desc="Processing jobs"):
        try:
            job = testads.iloc[idx]
            
            matches = match_job_batched(
                job['short_texts'], 
                esco_jobs, 
                batch_size=batch_size,
                verbose=True
            )

            print("\nMatching Results:")
            for i, match in enumerate(matches[:10], 1):
                print(f"\n{i}. ESCO ID: {match['esco_id']}")
                print(f"   Confidence: {match['confidence']}")
                print(f"   Reasoning: {match['reasoning']}")
            
            if not matches:
                continue
    
            evaluation = evaluate_match(matches, job['esco_id'])
            
            print("\nEvaluation:")
            for metric, value in evaluation.items():
                print(f"{metric}: {value}")
                
            result = {
                "model": MODEL,
                "job_id": job.name,
                "esco_id": job['esco_id'],
                "MRR": evaluation['mrr'],
                "missing": evaluation['missing'],
                "rank": evaluation['correct_rank'],
                "MRR@": MRR_AT,
                "top_20_hit": evaluation['top_20_hit'],
                "top_100_hit": evaluation['top_100_hit']
            }
            
            MRR.append(result)
            # Save after each job is processed
            results_df = pd.DataFrame(MRR)
            results_df.to_csv("../00_data/LLM_evaluation/llm_prompt_evaluation_results.csv", index=False)
            
        except Exception as e:
            print(f"Error processing job {idx}: {e}")
    
    print(f"\nFinal MRR@100 mean: {results_df['MRR'].mean():.3f}")
    return results_df

if __name__ == "__main__":
    # Process all jobs
    df = process_jobs(testads, esco_jobs, batch_size=50, number_of_jobs=250)
   
    # Save final results with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    df.to_csv(f"../00_data/LLM_evaluation/evaluation_results_{timestamp}.csv", index=False)