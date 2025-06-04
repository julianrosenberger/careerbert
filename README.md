# CareerBERT: Matching Resumes to ESCO Jobs in a Shared Embedding Space for Generic Job Recommendations

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016/j.eswa.2025.127043-blue)](https://doi.org/10.1016/j.eswa.2025.127043)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

This repository contains the source code and resources for the research paper "CareerBERT: Matching resumes to ESCO jobs in a shared embedding space for generic job recommendations," published in Expert Systems With Applications.

CareerBERT is an advanced support tool designed for career counselors and job seekers. It leverages Natural Language Processing (NLP) to match unstructured textual data from resumes to general job categories based on the European Skills, Competences, and Occupations (ESCO) taxonomy, augmented with EURopean Employment Services (EURES) job advertisements. The system aims to provide more accurate, comprehensive, and up-to-date job recommendations compared to traditional approaches.

## Key Features & Contributions

* **Innovative Corpus Creation**: Combines data from the ESCO taxonomy and EURES job advertisements for an accurate and current representation of the labor market.
* **Effective Domain Adaptation**: Demonstrates that domain-specific pre-training (using jobGBERT) significantly improves job recommendation performance over general language models and other state-of-the-art embedding approaches.
* **Comprehensive Evaluation**: Features a two-step evaluation process:
    * An application-grounded evaluation using EURES job advertisements.
    * A human-grounded evaluation with real-world resumes and feedback from HR experts.
* **Advanced Preprocessing**: Implements techniques like an internal relevance classifier for job advertisement shortening to reduce noise and enhance embedding quality.
* **Open Science**: Provides publicly available source code and pre-trained models to ensure reproducibility and encourage further research and application.

## Methodology Overview

CareerBERT utilizes a fine-tuned Sentence-BERT (SBERT) architecture to represent both resumes and jobs (derived from ESCO job classifications and EURES advertisements) within a shared high-dimensional embedding space. Job matching is performed by computing the cosine similarity between resume and job embeddings.

The system's development involved:
1.  **Data Preprocessing**:
    * Combining the structured ESCO taxonomy (for standardized job representations) with real-world EURES job advertisements (for up-to-date market trends).
    * Training a classifier to shorten job advertisements, focusing on relevant sections and mitigating BERT's token limit.
    * Creating "job centroids" by averaging embeddings from EURES job ads and ESCO job descriptions to represent job categories robustly.
2.  **Model Training**:
    * Utilizing German BERT (GBERT) and a domain-adapted version, jobGBERT, as base models.
    * Employing an SBERT Siamese network architecture fine-tuned with a Multiple Negatives Ranking (MNR) loss function.
    * Training data was constructed from ESCO by pairing job titles with their corresponding skills, descriptions, and synonyms.
    * The impact of task-adaptive pre-training using Transformer-based Sequential Denoising Auto-Encoder (TSDAE) was also investigated.

## Access Models & Code

* **Source Code**: The complete source code for experiments and models is available in this repository [(https://github.com/julianrosenberger/careerbert)](https://github.com/julianrosenberger/careerbert).
* **Pre-trained Models**:
    * `careerbert-jg` (jobGBERT base): [https://huggingface.co/lwolfrum2/careerbert-jg](https://huggingface.co/lwolfrum2/careerbert-jg)
    * `careerbert-g` (GBERT base): [https://huggingface.co/lwolfrum2/careerbert-g](https://huggingface.co/lwolfrum2/careerbert-g)

An example of a Streamlit application interface demonstrating CareerBERT's usage is described in Appendix A of the paper.

## Evaluation Highlights

* **Application-Grounded**:
    * CareerBERT (jobGBERT base) achieved an MRR@100 of 0.328, outperforming traditional methods (Word2Vec, Doc2Vec), conSultantBERT (reconstructed), ESCOXLM-R, and OpenAI's `text-embedding-ada-002`, and performing competitively with `text-embedding-3-small`.
    * Intelligent truncation of job advertisements using a classifier significantly improved matching performance.
    * Job advertisement centroids and hybrid job centroids (EURES + ESCO) proved effective for job representation.
* **Human-Grounded**:
    * Evaluated by 10 HR experts on 5 real-world resumes, CareerBERT (jobGBERT base with job centroids) demonstrated strong practical performance.
    * Achieved an average MAP@20 of 0.711, P@20 of 0.559, and MRR@20 of 0.861.
    * Showed versatility across diverse occupational domains, including blue-collar and atypical career profiles.
    * Performance was noted to be sensitive to the completeness of resume information.

## Citation

If you use CareerBERT or insights from our paper in your research, please cite:

```bibtex
@article{rosenberger2025careerbert,
  title = {CareerBERT: Matching resumes to ESCO jobs in a shared embedding space for generic job recommendations},
  author = {Julian Rosenberger and Lukas Wolfrum and Sven Weinzierl and Mathias Kraus and Patrick Zschech},
  journal = {Expert Systems With Applications},
  volume = {275},
  pages = {127043},
  year = {2025},
  doi = {[https://doi.org/10.1016/j.eswa.2025.127043](https://doi.org/10.1016/j.eswa.2025.127043)},
  issn = {0957-4174}
}
```

## Authors

* Julian Rosenberger (Universität Regensburg)
* Lukas Wolfrum (Friedrich-Alexander-Universität Erlangen-Nürnberg)
* Sven Weinzierl (Friedrich-Alexander-Universität Erlangen-Nürnberg)
* Mathias Kraus (Universität Regensburg)
* Patrick Zschech (TU Dresden)

## License

This work, as presented in the paper, is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).
