from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def setup_classification_pipeline():
    model_path = "../00_data/Classifier/model_classification_jobgbert/"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-classification", model, tokenizer=tokenizer)