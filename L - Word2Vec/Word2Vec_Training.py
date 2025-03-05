import numpy as np
import torch
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import re
from typing import List, Tuple, Dict
from helpers import *
from nltk.stem.snowball import SnowballStemmer

class ElsaftyDenseRecommender:
    """
    Implementation of Elsafty et al.'s Word2Vec and Doc2Vec approaches with:
    - Snowball stemming
    - Doc2Vec variants (DBOW and DM)
    - Word2Vec configurations from paper
    """
    
    def __init__(self, model_type: str = "word2vec", vector_size: int = 500):
        """
        Initialize recommender model
        
        Args:
            model_type: One of "word2vec", "doc2vec-dbow", or "doc2vec-dm"
            vector_size: Dimension of vectors (500 as per paper)
        """
        self.model_type = model_type
        self.vector_size = vector_size
        self.model = None
        self.stemmer = SnowballStemmer("german")  # German stemmer as per paper

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text following Elsafty's approach:
        - Remove HTML, URLs, emails
        - Remove special characters
        - Lowercase
        - Replace numbers
        - Apply stemming
        """
        text = re.sub(r'<[^>]+>', '', str(text))
        text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = re.sub(r'\d+', 'NUM', text)
        
        # Apply stemming
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def encode(self, texts, show_progress_bar=True):
        """Encode texts to vectors with consistent dtype"""
        vectors = []
        iterator = tqdm(texts) if show_progress_bar else texts
        
        for text in iterator:
            vec = self.get_document_vector(text)
            vectors.append(vec.astype(np.float32))
        return np.array(vectors, dtype=np.float32)

    def get_document_vector(self, text: str) -> np.ndarray:
        """Get vector representation for a document"""
        processed = self.preprocess_text(text)
        tokens = processed.split()
        
        if self.model_type == "word2vec":
            vectors = []
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            if vectors:
                return np.mean(vectors, axis=0)
            return np.zeros(self.vector_size)
        else:
            return self.model.infer_vector(tokens)

    def compute_similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity with consistent dtype handling"""
        embeddings1 = torch.tensor(embeddings1, dtype=torch.float32)
        embeddings2 = torch.tensor(embeddings2, dtype=torch.float32)
        
        embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        return torch.mm(embeddings1, embeddings2.t()).numpy()

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,
                   num_epochs: int = 20,  # As per paper
                   lr: float = 2e-5):
        """Train model and evaluate with job matching"""
        # Create directories
        os.makedirs(f"{output_path}/eval", exist_ok=True)
        os.makedirs(f"{output_path}_best", exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # Initialize tracking
        MRR = []
        MRR_AT = 100
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        max_MRR = 0
        best_model_to_save = None
        
        # Create train/dev split
        pos_train_samples, pos_dev_samples = train_test_split(pos_pairs, test_size=0.2, random_state=42)
        print(f"Training samples: {len(pos_train_samples)}, Development samples: {len(pos_dev_samples)}")
        
        # Prepare training data
        print("Preparing training data...")
        train_texts = []
        for text1, text2 in tqdm(pos_train_samples):
            train_texts.extend([self.preprocess_text(text1).split(), 
                              self.preprocess_text(text2).split()])

        # Train model based on type
        print("Training model...")
        if self.model_type == "word2vec":
            self.model = Word2Vec(
                sentences=train_texts,
                vector_size=self.vector_size,
                window=10,            # As per paper
                min_count=5,          # As per paper
                negative=15,          # As per paper
                epochs=num_epochs,    # As per paper
                workers=4
            )
        elif self.model_type == "doc2vec-dbow":
            # Doc2Vec with DBOW (as per paper)
            tagged_docs = [TaggedDocument(words=text, tags=[f'doc_{i}']) 
                         for i, text in enumerate(train_texts)]
            self.model = Doc2Vec(
                documents=tagged_docs,
                vector_size=self.vector_size,
                window=10,
                min_count=5,
                negative=15,
                epochs=num_epochs,
                dm=0,  # Use DBOW
                workers=4
            )
        else:  # doc2vec-dm
            # Doc2Vec with DM (as per paper)
            tagged_docs = [TaggedDocument(words=text, tags=[f'doc_{i}']) 
                         for i, text in enumerate(train_texts)]
            self.model = Doc2Vec(
                documents=tagged_docs,
                vector_size=self.vector_size,
                window=10,
                min_count=5,
                negative=15,
                epochs=num_epochs,
                dm=1,  # Use DM
                dm_mean=1,  # Use mean of context vectors
                workers=4
            )

        # Evaluate using ESCO embeddings
        print("Evaluating with ESCO embeddings...")
        testads = pd.DataFrame(load_json("../00_data/EURES/eures_testads_final_short.json"))
        
        print("Encoding test advertisements...")
        encodings_short = self.encode(testads["short_texts"])
        testads["embeddings_short"] = encodings_short.tolist()
        
        # Get job embeddings
        embeddings = encode_jobs(self)
        
        # Calculate similarities and MRR
        similarities = {}
        for k in embeddings:
            print(f"\nEvaluating {k} embeddings...")
            similarities[k] = self.compute_similarity(
                np.array(testads["embeddings_short"].tolist()),
                np.array(embeddings[k]["embeddings"])
            )
            
            ranks = []
            missing = 0
            simdf = pd.DataFrame(similarities[k], columns=embeddings[k]["esco_id"], index=testads["esco_id"])
            
            for i in tqdm(range(len(simdf)), desc="Calculating ranks"):
                id = simdf.iloc[i].name
                series = simdf.iloc[i].sort_values(ascending=False).reset_index()
                rank = (series[series["index"]==id].index.item()+1)
                
                if rank > MRR_AT:
                    missing += 1
                    ranks.append(0)
                else:
                    ranks.append(1/rank)
            
            missing = missing/len(simdf)
            current_run = {
                "model": output_path.split("/")[-1],
                "embedding_kind": k,
                "MRR": np.mean(ranks),
                "missing": missing,
                "MRR@": MRR_AT,
                "training_details": [training_start, batch_size, lr, num_epochs]
            }
            MRR.append(current_run)
            
            df = pd.DataFrame(MRR)
            print(f"\nResults for {k}:")
            print(df)
            
            # Save if best model
            if np.mean(ranks) > max_MRR:
                print(f"New best model saved (MRR: {np.mean(ranks):.4f})")
                max_MRR = np.mean(ranks)
                best_model_to_save = self.model
                model_path = os.path.join(f"{output_path}_best", 
                                        f"{self.model_type}.model")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                best_model_to_save.save(model_path)
                write_json(f"{output_path}/model_info.json", current_run)
            
            df.to_excel(f"{output_path}/eval/{training_start}_training_details.xlsx")
        
        # Save final model
        if best_model_to_save is not None:
            model_path = os.path.join(output_path, f"{self.model_type}.model")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            best_model_to_save.save(model_path)
            
        return MRR, best_model_to_save

if __name__ == "__main__":
    # Load data
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    print(f"Training with {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
    
    # Train Word2Vec model
    word2vec_model = ElsaftyDenseRecommender(model_type="word2vec")
    mrr_results_w2v, best_model_w2v = word2vec_model.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/Dense_Models/word2vec",
        num_epochs=20  # As per paper
    )
    
    # Train Doc2Vec DBOW model
    doc2vec_dbow = ElsaftyDenseRecommender(model_type="doc2vec-dbow")
    mrr_results_dbow, best_model_dbow = doc2vec_dbow.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/Dense_Models/doc2vec_dbow",
        num_epochs=20
    )
    
    # Train Doc2Vec DM model
    doc2vec_dm = ElsaftyDenseRecommender(model_type="doc2vec-dm")
    mrr_results_dm, best_model_dm = doc2vec_dm.train_model(
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        output_path="../00_data/Dense_Models/doc2vec_dm",
        num_epochs=20
    )
    
    # Print final results
    print("\nTraining completed")
    print("\nWord2Vec Results:")
    print(pd.DataFrame(mrr_results_w2v))
    print("\nDoc2Vec DBOW Results:")
    print(pd.DataFrame(mrr_results_dbow))
    print("\nDoc2Vec DM Results:")
    print(pd.DataFrame(mrr_results_dm))