from helpers import *
from datetime import datetime
import random

random.seed(42)
config = Config()

# Define models to evaluate
models_to_evaluate = [
    "openai-3-large",
    "jobgbert_batch32_woTSDAE_2e-05_f10/",  
    "openai-3-small",
    "openai-ada",
    "gbert_batch32_woTSDAE_2e-05_f10/",
]

def evaluate_single_model(model_path):
    """Evaluate a single model - extracted for clarity"""
    print(f"Loading Model {model_path}")
    
    # Use unified model creation
    model_info, embeddings = create_model_and_get_embeddings(model_path)
    
    # Load test data
    datasets = load_standard_datasets()
    testads = datasets["testads"]
    
    # Create test embeddings
    print("Creating embeddings for test ads.")
    encodings_short = encode_texts(list(testads["short_texts"]), model_info)
    testads["embeddings_short"] = encodings_short.tolist()
    
    # Evaluate using helper function
    return evaluate_embeddings(testads, embeddings, model_path)

def main():
    results = []
    for model_path in models_to_evaluate:
        result = evaluate_single_model(model_path)
        results.extend(result)
    
    # Save results
    df = pd.DataFrame(results).sort_values(by=["MRR"], ascending=[False])
    timestamp = "".join([c for c in str(datetime.now()).split('.')[0] if c.isdigit()])
    
    # Create output directory if it doesn't exist
    os.makedirs(config.EVALUATION_OUTPUT, exist_ok=True)
    output_path = f"{config.EVALUATION_OUTPUT}/{timestamp}_evaluation.xlsx"
    df.to_excel(output_path)
    
    print(f"\nResults saved to: {output_path}")
    print("\nEvaluation Results:")
    print(df)

if __name__ == "__main__":
    main()