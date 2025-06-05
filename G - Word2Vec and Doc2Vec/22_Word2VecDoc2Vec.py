import numpy as np
import torch
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import re
from typing import List, Tuple, Dict
import sys
sys.path.append('..')
from helpers import *  # Use your main comprehensive helpers
from nltk.stem.snowball import SnowballStemmer

class DenseEmbeddingModel:
    """
    Unified implementation for Word2Vec and Doc2Vec models
    Following Elsafty et al. approach with consistent interface to SBERT models
    """
    
    def __init__(self, model_type: str = "word2vec", vector_size: int = 500):
        """
        Initialize dense embedding model
        
        Args:
            model_type: One of "word2vec", "doc2vec-dbow", or "doc2vec-dm"
            vector_size: Dimension of vectors (500 as per Elsafty paper)
        """
        self.model_type = model_type
        self.vector_size = vector_size
        self.model = None
        self.stemmer = SnowballStemmer("german")
        
        # Validate model type
        valid_types = ["word2vec", "doc2vec-dbow", "doc2vec-dm"]
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text following Elsafty's approach with stemming
        Returns list of tokens for consistency with gensim
        """
        # Clean text
        text = re.sub(r'<[^>]+>', '', str(text))  # Remove HTML
        text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)  # Remove URLs/emails
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
        text = text.lower()
        text = re.sub(r'\d+', 'NUM', text)  # Replace numbers
        
        # Tokenize and stem
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words if len(word) > 2]
        return stemmed_words

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode texts to vectors - consistent interface with SentenceTransformer
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress
            
        Returns:
            numpy array of embeddings with shape (len(texts), vector_size)
        """
        vectors = []
        iterator = tqdm(texts, desc="Encoding texts") if show_progress_bar else texts
        
        for text in iterator:
            vec = self.get_document_vector(text)
            vectors.append(vec.astype(np.float32))
        
        return np.array(vectors, dtype=np.float32)

    def get_document_vector(self, text: str) -> np.ndarray:
        """Get vector representation for a single document"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
            
        tokens = self.preprocess_text(text)
        
        if self.model_type == "word2vec":
            # Average word vectors
            vectors = []
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            
            if vectors:
                return np.mean(vectors, axis=0).astype(np.float32)
            else:
                return np.zeros(self.vector_size, dtype=np.float32)
        else:
            # Doc2Vec inference
            return self.model.infer_vector(tokens).astype(np.float32)

    def train_model(self, pos_pairs: List[Tuple[str, str]], neg_pairs: List[Tuple[str, str]], 
                   output_path: str, **kwargs) -> Tuple[List[Dict], object]:
        """
        Train dense embedding model and evaluate with ESCO benchmark
        
        Args:
            pos_pairs: List of positive (anchor, positive) pairs
            neg_pairs: List of negative (anchor, negative) pairs  
            output_path: Path to save model and results
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (evaluation_results, best_model)
        """
        # Extract parameters with defaults
        num_epochs = kwargs.get('num_epochs', 20)  # As per Elsafty paper
        test_size = kwargs.get('test_size', 0.2)
        
        # Create directories
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        os.makedirs(f"{output_path}_best", exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # Initialize tracking
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        
        # Create train/dev split
        pos_train_samples, pos_dev_samples = train_test_split(
            pos_pairs, test_size=test_size, random_state=42
        )
        print(f"Training: {len(pos_train_samples)} pos pairs")
        print(f"Development: {len(pos_dev_samples)} pos pairs")
        
        # Prepare training corpus
        print("Preparing training corpus...")
        train_texts = []
        for text1, text2 in tqdm(pos_train_samples, desc="Processing training pairs"):
            train_texts.extend([
                self.preprocess_text(text1),
                self.preprocess_text(text2)
            ])

        # Train model based on type
        print(f"Training {self.model_type} model...")
        self._train_gensim_model(train_texts, num_epochs)
        
        # Evaluate using ESCO
        print("Evaluating with ESCO embeddings...")
        esco_results = self.evaluate_esco_metrics(output_path, training_start)
        
        # Save best model
        if esco_results:
            best_mrr = max(esco_results, key=lambda x: x['MRR'])['MRR']
            print(f"Best MRR: {best_mrr:.4f}")
            
            # Save best model
            model_path = f"{output_path}_best/{self.model_type}.model"
            self.model.save(model_path)
            
            # Save model info
            model_info = {
                "model_type": self.model_type,
                "best_mrr": best_mrr,
                "vector_size": self.vector_size,
                "epochs": num_epochs,
                "training_pairs": len(pos_train_samples)
            }
            write_json(f"{output_path}/model_info.json", model_info)
            
            # Save results
            results_df = pd.DataFrame(esco_results)
            results_df.to_excel(f"{output_path}/eval/{training_start}_{self.model_type}_results.xlsx")
        
        # Save final model
        final_model_path = f"{output_path}/{self.model_type}.model"
        self.model.save(final_model_path)
        
        return esco_results, self.model

    def _train_gensim_model(self, train_texts: List[List[str]], num_epochs: int):
        """Train the appropriate gensim model"""
        common_params = {
            'vector_size': self.vector_size,
            'window': 10,
            'min_count': 5,
            'negative': 15,
            'epochs': num_epochs,
            'workers': 4
        }
        
        if self.model_type == "word2vec":
            self.model = Word2Vec(sentences=train_texts, **common_params)
            
        elif self.model_type == "doc2vec-dbow":
            # Doc2Vec with DBOW
            tagged_docs = [
                TaggedDocument(words=text, tags=[f'doc_{i}']) 
                for i, text in enumerate(train_texts)
            ]
            self.model = Doc2Vec(
                documents=tagged_docs,
                dm=0,  # DBOW
                **common_params
            )
            
        elif self.model_type == "doc2vec-dm":
            # Doc2Vec with DM
            tagged_docs = [
                TaggedDocument(words=text, tags=[f'doc_{i}']) 
                for i, text in enumerate(train_texts)
            ]
            self.model = Doc2Vec(
                documents=tagged_docs,
                dm=1,  # DM
                dm_mean=1,  # Use mean of context vectors
                **common_params
            )

    def evaluate_esco_metrics(self, output_path: str, training_start: str) -> List[Dict]:
        """Evaluate using ESCO embeddings - consistent with SBERT evaluation"""
        try:
            # Load test data using main helpers
            testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
            
            # Encode test ads
            print("Encoding test advertisements...")
            encodings_short = self.encode(list(testads["short_texts"]), show_progress_bar=True)
            testads["embeddings_short"] = encodings_short.tolist()
            
            # Generate ESCO embeddings using main encode_jobs function
            embeddings = encode_jobs(self)
            
            MRR = []
            MRR_AT = 100
            
            # Evaluate each embedding type
            for k in embeddings:
                print(f"Evaluating {k} embeddings...")
                similarities = util.cos_sim(
                    testads["embeddings_short"], 
                    embeddings[k]["embeddings"]
                )
                
                ranks = []
                missing = 0
                simdf = pd.DataFrame(
                    similarities, 
                    columns=embeddings[k]["esco_id"], 
                    index=testads["esco_id"]
                )
                
                for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
                    id = simdf.iloc[i].name
                    series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                    rank = (series[series["index"] == id].index.item() + 1)
                    
                    if rank > MRR_AT:
                        missing += 1
                        ranks.append(0)
                    else:
                        ranks.append(1/rank)
                
                missing = missing / len(simdf)
                
                result = {
                    "model": f"{output_path.split('/')[-1]}_{self.model_type}",
                    "embedding_kind": k,
                    "MRR": np.mean(ranks),
                    "missing": missing,
                    "MRR@": MRR_AT,
                    "mode": self.model_type,
                    "training_timestamp": training_start
                }
                MRR.append(result)
                
                print(f"MRR for {k}: {np.mean(ranks):.4f}")
            
            return MRR
            
        except Exception as e:
            print(f"ESCO evaluation failed: {e}")
            return []

    @classmethod
    def load_trained_model(cls, model_path: str, model_type: str) -> 'DenseEmbeddingModel':
        """Load a pre-trained model"""
        instance = cls(model_type=model_type)
        
        if model_type == "word2vec":
            instance.model = Word2Vec.load(model_path)
        else:
            instance.model = Doc2Vec.load(model_path)
            
        return instance

def train_all_dense_models(pos_pairs: List[Tuple[str, str]], neg_pairs: List[Tuple[str, str]], 
                          base_output_path: str, use_subset: bool = True) -> List[Dict]:
    """
    Train all dense embedding models for comparison with ConsultantBERT
    
    Args:
        pos_pairs: Positive training pairs
        neg_pairs: Negative training pairs
        base_output_path: Base path for saving models
        use_subset: Whether to use subset for faster training
        
    Returns:
        List of all evaluation results
    """
    print("="*80)
    print("TRAINING ALL DENSE EMBEDDING MODELS")
    print("="*80)
    
    if use_subset:
        # Use subset for faster testing
        pos_subset = pos_pairs[:1000] if len(pos_pairs) > 1000 else pos_pairs
        neg_subset = neg_pairs[:1000] if len(neg_pairs) > 1000 else neg_pairs
        print(f"Using subset: {len(pos_subset)} pos, {len(neg_subset)} neg pairs")
        pos_pairs, neg_pairs = pos_subset, neg_subset
    
    models_to_train = ["word2vec", "doc2vec-dbow", "doc2vec-dm"]
    all_results = []
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        model = DenseEmbeddingModel(model_type=model_type)
        output_path = f"{base_output_path}_{model_type}"
        
        results_list, best_model = model.train_model(
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            output_path=output_path,
            num_epochs=20  # As per Elsafty paper
        )
        
        if results_list:
            all_results.extend(results_list)
            
            # Print summary for this model
            best_mrr = max(results_list, key=lambda x: x['MRR'])['MRR']
            print(f"Best MRR for {model_type}: {best_mrr:.4f}")
    
    # Create comparison summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Group by model and get best MRR for each
        summary = results_df.groupby('mode')['MRR'].max().reset_index()
        summary = summary.sort_values('MRR', ascending=False)
        
        print(f"\n{'='*60}")
        print("DENSE MODELS RESULTS SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Best MRR':<10}")
        print("-"*30)
        for _, row in summary.iterrows():
            print(f"{row['mode']:<20} {row['MRR']:<10.4f}")
        
        # Save combined results
        timestamp = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        results_df.to_excel(f"{base_output_path}_all_dense_models_{timestamp}.xlsx")
    
    return all_results

def evaluate_pretrained_dense_models(models_dir: str = "../00_data/Dense_Models") -> List[Dict]:
    """
    Evaluate pre-trained dense models for comparison
    
    Args:
        models_dir: Directory containing pre-trained models
        
    Returns:
        List of evaluation results
    """
    print("="*80)
    print("EVALUATING PRE-TRAINED DENSE MODELS")
    print("="*80)
    
    # Define model paths
    model_configs = [
        (f"{models_dir}/word2vec_best/word2vec.model", "word2vec"),
        (f"{models_dir}/doc2vec_dbow_best/doc2vec-dbow.model", "doc2vec-dbow"),
        (f"{models_dir}/doc2vec_dm_best/doc2vec-dm.model", "doc2vec-dm")
    ]
    
    all_results = []
    
    for model_path, model_type in model_configs:
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_type} from {model_path}")
            
            try:
                # Load pre-trained model
                model = DenseEmbeddingModel.load_trained_model(model_path, model_type)
                
                # Evaluate
                training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
                results = model.evaluate_esco_metrics(
                    output_path=os.path.dirname(model_path),
                    training_start=training_start
                )
                
                if results:
                    all_results.extend(results)
                    best_mrr = max(results, key=lambda x: x['MRR'])['MRR']
                    print(f"Best MRR for {model_type}: {best_mrr:.4f}")
                    
            except Exception as e:
                print(f"Error evaluating {model_path}: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    return all_results

def main():
    """Main function for dense model training and evaluation"""
    
    # Load data using main helpers
    print("Loading training data...")
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    print(f"Loaded {len(pos_pairs)} positive and {len(neg_pairs)} negative pairs")
    
    base_output_path = "../00_data/Dense_Models/dense_multilingual"
    
    # Option 1: Train new models
    training_results = train_all_dense_models(
        pos_pairs, neg_pairs, base_output_path, use_subset=True
    )
    
    # Option 2: Evaluate existing models
    # evaluation_results = evaluate_pretrained_dense_models()
    
    return training_results

if __name__ == "__main__":
    results = main()