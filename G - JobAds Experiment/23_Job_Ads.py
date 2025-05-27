import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pypdf import PdfReader
import json
from helpers import *

# Load configuration and datasets
config = Config()
datasets = load_standard_datasets()
testads = datasets["testads"]

def load_single_cv(filepath):
    """Load and extract text from a single CV PDF file"""
    cv = ""
    reader = PdfReader(filepath)
    pages = reader.pages
    for i in range(len(pages)):
        page = reader.pages[i].extract_text().strip()
        cv += page
    return cv

def load_cvs(cv_directory="../00_data/CVs/", cv_count=5):
    """Load multiple CVs with error handling"""
    cvs = []
    cv_texts = []
    
    for i in range(1, cv_count + 1):
        filepath = f"{cv_directory}CV_{i}.pdf"
        try:
            cv_text = load_single_cv(filepath)
            cvs.append({'id': f'CV_{i}', 'text': cv_text})
            cv_texts.append(cv_text)
            print(f"Successfully loaded {filepath}")
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return cvs, cv_texts

def load_analysis_model():
    """Load the model for CV-job analysis"""
    model_path = f"{config.MODELS_BASE}/jobgbert_batch32_woTSDAE_2e-05_f10/"
    print(f"Loading model from: {model_path}")
    return SentenceTransformer(model_path)

def analyze_cross_similarities():
    """Analyze similarities between CVs and job ads"""
    # Load model
    model = load_analysis_model()
    
    # Load CVs
    cvs, cv_texts = load_cvs()
    
    # Create embeddings for both job ads and CVs
    print("Creating embeddings...")
    job_embeddings = model.encode(list(testads["short_texts"]), show_progress_bar=True)
    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)
    
    # Calculate similarities between CVs and job ads
    print("Calculating CV-job similarities...")
    cv_job_similarities = util.cos_sim(cv_embeddings, job_embeddings)
    
    # For each CV, find most similar job ads
    results = []
    for i in range(len(cvs)):
        # Get similarities for current CV
        sim_scores = cv_job_similarities[i].numpy()
        
        # Get top 5 most similar jobs
        top_indices = np.argsort(sim_scores)[-5:][::-1]
        
        results.append({
            'cv_id': cvs[i]['id'],
            'cv_text': cvs[i]['text'][:200] + "...",  # First 200 chars
            'similar_jobs': [{
                'esco_id': testads.iloc[idx]['esco_id'],
                'similarity': float(sim_scores[idx]),
                'text': testads.iloc[idx]['short_texts'][:200] + "..."
            } for idx in top_indices]
        })
    
    # Calculate similarities between jobs with same ESCO ID
    print("Analyzing job-job similarities...")
    job_job_similarities = util.cos_sim(job_embeddings, job_embeddings)
    
    # Calculate average similarity between jobs with same ESCO ID
    same_esco_sims = []
    diff_esco_sims = []
    
    for i in range(len(testads)):
        for j in range(i + 1, len(testads)):
            sim = job_job_similarities[i][j].item()
            if testads.iloc[i]['esco_id'] == testads.iloc[j]['esco_id']:
                same_esco_sims.append(sim)
            else:
                diff_esco_sims.append(sim)
    
    similarity_stats = {
        'avg_same_esco_sim': np.mean(same_esco_sims),
        'avg_diff_esco_sim': np.mean(diff_esco_sims),
        'cv_job_avg_sim': float(cv_job_similarities.mean()),
        'cv_job_max_sim': float(cv_job_similarities.max())
    }
    
    return pd.DataFrame(results), similarity_stats

def calculate_similarity_distributions():
    """Calculate similarity distributions for visualization"""
    # Load model
    model = load_analysis_model()
    
    # Load CVs
    cvs, cv_texts = load_cvs()
    
    # Create embeddings
    print("Creating embeddings...")
    job_embeddings = model.encode(list(testads["short_texts"]), show_progress_bar=True)
    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)
    
    # Calculate similarity matrices
    print("Calculating similarities...")
    job_job_similarities = util.cos_sim(job_embeddings, job_embeddings)
    cv_job_similarities = util.cos_sim(cv_embeddings, job_embeddings)
    
    # Extract similarity distributions
    same_esco_sims = []
    diff_esco_sims = []
    job_job_sims = []
    cv_job_sims = cv_job_similarities.numpy().flatten()
    
    # Calculate job-job similarities
    for i in range(len(testads)):
        for j in range(i + 1, len(testads)):
            sim = job_job_similarities[i][j].item()
            job_job_sims.append(sim)
            if testads.iloc[i]['esco_id'] == testads.iloc[j]['esco_id']:
                same_esco_sims.append(sim)
            else:
                diff_esco_sims.append(sim)
    
    # Create bins for similarity scores
    bins = np.arange(0, 1.1, 0.1)
    
    # Calculate histograms
    job_job_hist, _ = np.histogram(job_job_sims, bins=bins, density=True)
    cv_job_hist, _ = np.histogram(cv_job_sims, bins=bins, density=True)
    same_esco_hist, _ = np.histogram(same_esco_sims, bins=bins, density=True)
    
    # Prepare distribution data
    distribution_data = []
    for i in range(len(bins)-1):
        distribution_data.append({
            'similarity': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            'jobJob': float(job_job_hist[i]),
            'jobResume': float(cv_job_hist[i]),
            'sameEsco': float(same_esco_hist[i])
        })
    
    # Calculate summary statistics
    stats = {
        'job_job_mean': np.mean(job_job_sims),
        'job_job_std': np.std(job_job_sims),
        'cv_job_mean': np.mean(cv_job_sims),
        'cv_job_std': np.std(cv_job_sims),
        'same_esco_mean': np.mean(same_esco_sims),
        'same_esco_std': np.std(same_esco_sims),
        'diff_esco_mean': np.mean(diff_esco_sims),
        'diff_esco_std': np.std(diff_esco_sims)
    }
    
    return distribution_data, stats

def main():
    """Main analysis function"""
    print("Starting CV-Job Analysis...")
    print("=" * 50)
    
    # Run cross-similarity analysis
    print("\n1. Running cross-similarity analysis...")
    results_df, stats = analyze_cross_similarities()
    
    # Print summary statistics
    print("\nSimilarity Statistics:")
    print(f"Average similarity between jobs with same ESCO ID: {stats['avg_same_esco_sim']:.3f}")
    print(f"Average similarity between jobs with different ESCO ID: {stats['avg_diff_esco_sim']:.3f}")
    print(f"Average CV-Job similarity: {stats['cv_job_avg_sim']:.3f}")
    print(f"Maximum CV-Job similarity: {stats['cv_job_max_sim']:.3f}")
    
    # Display example matches
    print("\nExample CV-Job Matches:")
    for _, row in results_df.iterrows():
        print(f"\nCV {row['cv_id']}:")
        print(f"Top similar jobs:")
        for job in row['similar_jobs'][:3]:  # Show top 3 matches
            print(f"- ESCO ID: {job['esco_id']}, Similarity: {job['similarity']:.3f}")
    
    # Run distribution analysis
    print("\n2. Calculating similarity distributions...")
    distributions, dist_stats = calculate_similarity_distributions()
    
    print("\nSimilarity Distribution Statistics:")
    print(f"Job-Job Similarities: mean={dist_stats['job_job_mean']:.3f} ± {dist_stats['job_job_std']:.3f}")
    print(f"Job-Resume Similarities: mean={dist_stats['cv_job_mean']:.3f} ± {dist_stats['cv_job_std']:.3f}")
    print(f"Same ESCO Similarities: mean={dist_stats['same_esco_mean']:.3f} ± {dist_stats['same_esco_std']:.3f}")
    print(f"Different ESCO Similarities: mean={dist_stats['diff_esco_mean']:.3f} ± {dist_stats['diff_esco_std']:.3f}")
    
    # Save results
    print("\n3. Saving results...")
    
    # Save CV-job matches
    results_df.to_json('cv_job_matches.json', orient='records', indent=2)
    print("CV-job matches saved to: cv_job_matches.json")
    
    # Save similarity statistics
    with open('similarity_stats.json', 'w') as f:
        json.dump({**stats, **dist_stats}, f, indent=2)
    print("Similarity statistics saved to: similarity_stats.json")
    
    # Save distribution data for visualization
    with open('similarity_distributions.json', 'w') as f:
        json.dump(distributions, f, indent=2)
    print("Distribution data saved to: similarity_distributions.json")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()