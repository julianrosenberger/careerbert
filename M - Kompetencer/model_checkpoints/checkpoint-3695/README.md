---
base_model: jjzha/esco-xlm-roberta-large
library_name: sentence-transformers
metrics:
- map
- mrr@100
- ndcg@100
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:118210
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: Techniker Automatisierungstechnik Technikerin Automatisierungstechnik
  sentences:
  - Unterricht für Schüler/Schülerinnen mit sonderpädagogischem Förderbedarf anbieten
  - Qualität von Erzeugnissen kontrollieren
  - Kinematografie
- source_sentence: Referent Referentin
  sentences:
  - Sanitätsfachverkäufer
  - wissenschaftliche Forschung
  - Problemlösungen finden
- source_sentence: Produktprüfer - Bekleidung Produktprüferin - Bekleidung
  sentences:
  - CAE Software
  - Finanzmärkte
  - Bekleidungsprodukte untersuchen
- source_sentence: Brennschneider Brennschneiderin
  sentences:
  - verschiedene Kommunikationskanäle verwenden
  - statistische Methoden der Prozesssteuerung anwenden
  - Schuhstepperin
- source_sentence: Bioingenieur Bioingenieurin
  sentences:
  - Selbstvertrauen zeigen
  - Maßnahmen zur Prävention von Fischkrankheiten ergreifen
  - analytisch-mathematische Berechnungen durchführen
model-index:
- name: SentenceTransformer based on jjzha/esco-xlm-roberta-large
  results:
  - task:
      type: reranking
      name: Reranking
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: map
      value: 0.6214999074568522
      name: Map
    - type: mrr@100
      value: 0.8377145701644096
      name: Mrr@100
    - type: ndcg@100
      value: 0.7878389163140403
      name: Ndcg@100
    - type: map
      value: 0.6701972636083754
      name: Map
    - type: mrr@100
      value: 0.8620417211957565
      name: Mrr@100
    - type: ndcg@100
      value: 0.8176961057954187
      name: Ndcg@100
    - type: map
      value: 0.7057606377271075
      name: Map
    - type: mrr@100
      value: 0.8738905683945934
      name: Mrr@100
    - type: ndcg@100
      value: 0.8394022681840605
      name: Ndcg@100
    - type: map
      value: 0.7174156615345353
      name: Map
    - type: mrr@100
      value: 0.8862501546482106
      name: Mrr@100
    - type: ndcg@100
      value: 0.8470683796224638
      name: Ndcg@100
    - type: map
      value: 0.7443401481686821
      name: Map
    - type: mrr@100
      value: 0.8932851817366837
      name: Mrr@100
    - type: ndcg@100
      value: 0.8618425244802297
      name: Ndcg@100
    - type: map
      value: 0.7647399902082354
      name: Map
    - type: mrr@100
      value: 0.900908394471564
      name: Mrr@100
    - type: ndcg@100
      value: 0.8737493082362766
      name: Ndcg@100
    - type: map
      value: 0.7746225490118622
      name: Map
    - type: mrr@100
      value: 0.9116362737534083
      name: Mrr@100
    - type: ndcg@100
      value: 0.8812498387532118
      name: Ndcg@100
    - type: map
      value: 0.7755264835676008
      name: Map
    - type: mrr@100
      value: 0.9099838218069742
      name: Mrr@100
    - type: ndcg@100
      value: 0.8807011627223548
      name: Ndcg@100
    - type: map
      value: 0.7946126353146487
      name: Map
    - type: mrr@100
      value: 0.918707973522715
      name: Mrr@100
    - type: ndcg@100
      value: 0.8918978180768041
      name: Ndcg@100
---

# SentenceTransformer based on jjzha/esco-xlm-roberta-large

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [jjzha/esco-xlm-roberta-large](https://huggingface.co/jjzha/esco-xlm-roberta-large). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [jjzha/esco-xlm-roberta-large](https://huggingface.co/jjzha/esco-xlm-roberta-large) <!-- at revision 7ffea8f23a422feb00de3ae7f3f7bcba21c15768 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: XLMRobertaModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Bioingenieur Bioingenieurin',
    'analytisch-mathematische Berechnungen durchführen',
    'Maßnahmen zur Prävention von Fischkrankheiten ergreifen',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.6215** |
| mrr@100  | 0.8377     |
| ndcg@100 | 0.7878     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.6702** |
| mrr@100  | 0.862      |
| ndcg@100 | 0.8177     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7058** |
| mrr@100  | 0.8739     |
| ndcg@100 | 0.8394     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7174** |
| mrr@100  | 0.8863     |
| ndcg@100 | 0.8471     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7443** |
| mrr@100  | 0.8933     |
| ndcg@100 | 0.8618     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7647** |
| mrr@100  | 0.9009     |
| ndcg@100 | 0.8737     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7746** |
| mrr@100  | 0.9116     |
| ndcg@100 | 0.8812     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7755** |
| mrr@100  | 0.91       |
| ndcg@100 | 0.8807     |

#### Reranking

* Evaluated with [<code>RerankingEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.RerankingEvaluator)

| Metric   | Value      |
|:---------|:-----------|
| **map**  | **0.7946** |
| mrr@100  | 0.9187     |
| ndcg@100 | 0.8919     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 118,210 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 3 tokens</li><li>mean: 12.78 tokens</li><li>max: 46 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 9.69 tokens</li><li>max: 142 tokens</li></ul> |
* Samples:
  | sentence_0                                                                 | sentence_1                                                      |
  |:---------------------------------------------------------------------------|:----------------------------------------------------------------|
  | <code>Statistischer Assistent Statistische Assistentin</code>              | <code>öffentliche Erhebungen durchführen</code>                 |
  | <code>Ingenieur für Nanotechnologie Ingenieurin für Nanotechnologie</code> | <code>Ingenieur für Nanotechnologie</code>                      |
  | <code>Akademischer Krankenpfleger Akademische Krankenschwester</code>      | <code>Forschung zur akademischen Krankenpflege betreiben</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | map    |
|:------:|:----:|:-------------:|:------:|
| 0.1353 | 500  | 3.2951        | -      |
| 0.2706 | 1000 | 2.2876        | -      |
| 0.4060 | 1500 | 1.9608        | -      |
| 0.5413 | 2000 | 1.7782        | -      |
| 0.6766 | 2500 | 1.655         | -      |
| 0.8119 | 3000 | 1.5531        | -      |
| 0.9472 | 3500 | 1.4938        | -      |
| 1.0    | 3695 | -             | 0.6215 |
| 0.1353 | 500  | 1.3575        | -      |
| 0.2706 | 1000 | 1.3268        | -      |
| 0.4060 | 1500 | 1.285         | -      |
| 0.5413 | 2000 | 1.2568        | -      |
| 0.6766 | 2500 | 1.2619        | -      |
| 0.8119 | 3000 | 1.2281        | -      |
| 0.9472 | 3500 | 1.2337        | -      |
| 1.0    | 3695 | -             | 0.6702 |
| 0.1353 | 500  | 1.1818        | -      |
| 0.2706 | 1000 | 1.1363        | -      |
| 0.4060 | 1500 | 1.111         | -      |
| 0.5413 | 2000 | 1.117         | -      |
| 0.6766 | 2500 | 1.1056        | -      |
| 0.8119 | 3000 | 1.1053        | -      |
| 0.9472 | 3500 | 1.0997        | -      |
| 1.0    | 3695 | -             | 0.7058 |
| 0.1353 | 500  | 1.057         | -      |
| 0.2706 | 1000 | 1.0269        | -      |
| 0.4060 | 1500 | 0.9919        | -      |
| 0.5413 | 2000 | 0.9983        | -      |
| 0.6766 | 2500 | 1.0031        | -      |
| 0.8119 | 3000 | 1.0195        | -      |
| 0.9472 | 3500 | 1.0108        | -      |
| 1.0    | 3695 | -             | 0.7174 |
| 0.1353 | 500  | 1.0138        | -      |
| 0.2706 | 1000 | 0.9529        | -      |
| 0.4060 | 1500 | 0.9326        | -      |
| 0.5413 | 2000 | 0.9396        | -      |
| 0.6766 | 2500 | 0.9426        | -      |
| 0.8119 | 3000 | 0.9521        | -      |
| 0.9472 | 3500 | 0.9634        | -      |
| 1.0    | 3695 | -             | 0.7443 |
| 0.1353 | 500  | 0.9139        | -      |
| 0.2706 | 1000 | 0.9071        | -      |
| 0.4060 | 1500 | 0.8666        | -      |
| 0.5413 | 2000 | 0.8847        | -      |
| 0.6766 | 2500 | 0.884         | -      |
| 0.8119 | 3000 | 0.9035        | -      |
| 0.9472 | 3500 | 0.8949        | -      |
| 1.0    | 3695 | -             | 0.7647 |
| 0.1353 | 500  | 0.8685        | -      |
| 0.2706 | 1000 | 0.8395        | -      |
| 0.4060 | 1500 | 0.8358        | -      |
| 0.5413 | 2000 | 0.8319        | -      |
| 0.6766 | 2500 | 0.8437        | -      |
| 0.8119 | 3000 | 0.8627        | -      |
| 0.9472 | 3500 | 0.8703        | -      |
| 1.0    | 3695 | -             | 0.7746 |
| 0.1353 | 500  | 0.8271        | -      |
| 0.2706 | 1000 | 0.7998        | -      |
| 0.4060 | 1500 | 0.7932        | -      |
| 0.5413 | 2000 | 0.7925        | -      |
| 0.6766 | 2500 | 0.8159        | -      |
| 0.8119 | 3000 | 0.8206        | -      |
| 0.9472 | 3500 | 0.8192        | -      |
| 1.0    | 3695 | -             | 0.7755 |
| 0.1353 | 500  | 0.7895        | -      |
| 0.2706 | 1000 | 0.768         | -      |
| 0.4060 | 1500 | 0.7612        | -      |
| 0.5413 | 2000 | 0.7778        | -      |
| 0.6766 | 2500 | 0.7723        | -      |
| 0.8119 | 3000 | 0.7967        | -      |
| 0.9472 | 3500 | 0.8043        | -      |
| 1.0    | 3695 | -             | 0.7946 |
| 0.1353 | 500  | 0.7664        | -      |
| 0.2706 | 1000 | 0.7523        | -      |
| 0.4060 | 1500 | 0.7323        | -      |
| 0.5413 | 2000 | 0.7436        | -      |
| 0.6766 | 2500 | 0.7468        | -      |
| 0.8119 | 3000 | 0.7585        | -      |
| 0.9472 | 3500 | 0.766         | -      |


### Framework Versions
- Python: 3.10.16
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu124
- Accelerate: 1.2.1
- Datasets: 2.19.1
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->