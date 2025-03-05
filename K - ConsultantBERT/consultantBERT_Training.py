from sentence_transformers import SentenceTransformer, InputExample, models, losses, util, evaluation
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from helpers import *

class ConsultantBERT:
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased", 
                 max_seq_length: int = 512, 
                 mode: str = "regressor"):
        """
        Initialize ConsultantBERT model
        
        Args:
            model_name: Base BERT model to use
            max_seq_length: Maximum sequence length
            mode: Either "classifier" or "regressor"
        """
        set_seed(42)
        self.mode = mode
        
        # Create word embedding model
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        
        # Initialize SBERT with appropriate modules
        self.sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # Store embedding dimension
        self.embedding_dim = word_embedding_model.get_word_embedding_dimension()

    def train_model(self, pos_pairs, neg_pairs, output_path: str,
                   batch_size: int = 4,
                   num_epochs: int = 5,
                   lr: float = 2e-5,
                   evaluation_steps: int = 1000):
        """Train model using either classification or regression objective"""
        
        # Initialize tracking
        MRR = []
        MRR_AT = 100
        training_start = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
        max_MRR = 0
        best_model = None
        
        # Create train/dev split
        pos_train_samples, pos_dev_samples = train_test_split(pos_pairs, test_size=0.2, random_state=42)
        print(f"Training samples: {len(pos_train_samples)}, Development samples: {len(pos_dev_samples)}")
        
        # Create development set
        dev_set = create_training_samples(pos_dev_samples, neg_pairs)
        print("Created development set")
        
        # Prepare training examples
        train_examples = []
        if self.mode == "classifier":
            # Create a dictionary for faster negative pair lookup
            print("Creating negative pairs dictionary...")
            neg_pairs_dict = {}
            for anchor, negative in neg_pairs:
                if anchor not in neg_pairs_dict:
                    neg_pairs_dict[anchor] = []
                neg_pairs_dict[anchor].append(negative)
            
            # Create balanced dataset with positive and negative pairs
            print("Creating training examples for classifier mode...")
            for idx, (anchor, positive) in enumerate(tqdm(pos_train_samples)):
                # Add positive example (label 1)
                train_examples.append(InputExample(
                    texts=[anchor, positive],
                    label=1  # Positive pair
                ))
                
                # Get negatives for this anchor (limit to 1 negative per positive)
                if anchor in neg_pairs_dict and neg_pairs_dict[anchor]:
                    train_examples.append(InputExample(
                        texts=[anchor, neg_pairs_dict[anchor][0]],
                        label=0  # Negative pair
                    ))
                
                # Print progress every 10000 samples
                if idx % 10000 == 0:
                    print(f"Processed {idx}/{len(pos_train_samples)} samples")
        else:
            print("Creating training examples for regressor mode")
            # For regressor mode, use simple pairs
            for item in pos_train_samples:
                train_examples.append(InputExample(
                    texts=[item[0], item[1]]
                ))
        
        print(f"Created {len(train_examples)} training examples")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps
        warmup_steps = int(len(train_examples) * num_epochs * 0.1)
        
        # Set up loss based on mode
        if self.mode == "classifier":
            train_loss = losses.SoftmaxLoss(
                model=self.sbert_model,
                sentence_embedding_dimension=self.embedding_dim,
                num_labels=2  # Binary classification
            )
        else:  # regressor
            train_loss = losses.MultipleNegativesRankingLoss(self.sbert_model)
            
        print("Created loss function")
            
        # Set up evaluator
        evaluator = evaluation.RerankingEvaluator(dev_set, name='dev', batch_size=batch_size)
        print("Created evaluator")
        
        print("Starting training...")
        # Train the model
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            optimizer_params={'lr': lr},
            show_progress_bar=True
        )
        
        # Evaluate model
        mrr_score = self._evaluate_mrr(pos_dev_samples, neg_pairs)
        
        # Save best model
        if mrr_score > max_MRR:
            print(f"New best model saved with MRR: {mrr_score}")
            max_MRR = mrr_score
            self.sbert_model.save(f"{output_path}_best")
            
        return MRR, self.sbert_model

    def _evaluate_mrr(self, test_samples, neg_pairs, mrr_at: int = 100):
        """Calculate Mean Reciprocal Rank"""
        ranks = []
        for query, positive in tqdm(test_samples):
            # Get negative samples
            negatives = [neg[1] for neg in neg_pairs if neg[0] == query]
            
            # Get embeddings
            query_embedding = self.sbert_model.encode([query], convert_to_tensor=True)
            positive_embedding = self.sbert_model.encode([positive], convert_to_tensor=True)
            negative_embeddings = self.sbert_model.encode(negatives, convert_to_tensor=True)
            
            # Calculate similarities
            pos_sim = util.pytorch_cos_sim(query_embedding, positive_embedding)
            neg_sims = util.pytorch_cos_sim(query_embedding, negative_embeddings)
            
            # Calculate rank
            all_sims = torch.cat([pos_sim, neg_sims], dim=1)
            sorted_sims, indices = torch.sort(all_sims, descending=True)
            rank = (indices == 0).nonzero().item() + 1
            
            if rank <= mrr_at:
                ranks.append(1.0/rank)
            else:
                ranks.append(0)
                
        return np.mean(ranks)

def create_training_samples(pos_dev_samples, neg_pairs):
    """Create development set with query/positive/negative structure"""
    dev_set = []
    anchors = set([x[0] for x in pos_dev_samples])
    neg_dev_samples = [x for x in neg_pairs if x[0] in anchors]
    
    print("Creating development set")
    for anchor in tqdm(anchors):
        pos_pairs = [x[1] for x in pos_dev_samples if x[0]==anchor]
        neg_pairs = [x[1] for x in neg_dev_samples if x[0]==anchor]
        dev_set.append({
            "query": anchor,
            "positive": pos_pairs,
            "negative": neg_pairs
        })
    return dev_set

if __name__ == "__main__":
    # Load data
    data_dict = load_data_pairs()
    pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
    neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])
    
    # For classifier version
    model_classifier = ConsultantBERT(mode="classifier")
    mrr_cls, best_model_cls = model_classifier.train_model(pos_pairs, neg_pairs, "output_path_classifier")

    # For regressor version  
    model_regressor = ConsultantBERT(mode="regressor")
    mrr_reg, best_model_reg = model_regressor.train_model(pos_pairs, neg_pairs, "output_path_regressor")