# Evaluation of Greek-language LLM behavior for sentiment classification and explanation generation under prompt variation 

> MSc Thesis — Sentiment classification and natural-language explanations for Greek movie reviews, using open Greek LLMs (**Meltemi-7B** and **Llama-Krikri-8B**).

## Overview

This repository contains the full experimental pipeline for my MSc thesis on **explainable sentiment analysis for Greek**. The goal is not only to predict whether a movie review is **positive (θετικό)** or **negative (αρνητικό)**, but also to have the model produce a short **natural-language explanation** justifying its decision — and then to rigorously evaluate how *good* and how *faithful* those explanations are.

Because large, high-quality Greek sentiment datasets with explanations do not exist, the project builds its own data by sampling the English **IMDb** movie-review dataset and **machine-translating it into Greek**. Two open, Greek-adapted instruction-tuned LLMs are then studied under three complementary regimes:

1. **Few-shot in-context learning** (0–16 shots) using quantized GGUF models.
2. **Parameter-efficient fine-tuning** (QLoRA) on the translated data.
3. **Explanation quality & faithfulness evaluation** using an LLM-as-a-judge and a token-masking test.

### Models studied

| Role | Model | Source |
|------|-------|--------|
| Greek LLM #1 | **Meltemi-7B-Instruct v1.5** | [`ilsp/Meltemi-7B-Instruct-v1.5`](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5) |
| Greek LLM #2 | **Llama-Krikri-8B-Instruct** | [`ilsp/Llama-Krikri-8B-Instruct`](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct) |
| Machine translation (EN→EL) | **MADLAD-400-3B-MT** | [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) |
| Explanation judge | **Gemini 2.5 Pro** | Google Generative AI API |

---

## Task variants

Every experiment is run across three prompting/output variants, which let us test whether *reasoning order* affects sentiment accuracy and explanation quality:

| Variant | Model is asked to produce |
|---------|---------------------------|
| `sentiment_only` | Only the sentiment label |
| `sentiment_then_explanation` | Sentiment first, then the explanation |
| `explanation_then_sentiment` | Explanation first, then the sentiment (chain-of-thought style) |

All prompts are written **in Greek** and request a fixed output format (`Συναίσθημα:` / `Εξήγηση:`) to make parsing robust.

---

## Methodology / Pipeline

The scripts are meant to be run in the following order. Each was developed as a Google Colab notebook and exported to `.py`.

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Data preparation                                                 │
│    Sampling & Machine Translation                                   │
│    IMDb (50k, EN) → balanced samples → translate to Greek (MADLAD)  │
└─────────────────────────────────────────────────────────────────────┘
              │  sample_training_data_{1000,2000}.csv, test set
              ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ 2a. Few-shot baselines       │   │ 2b. Fine-tuning track            │
│  Meltemi GGUF / Krikri GGUF  │   │  Training Explanations Generation │
│  0,2,4,6,8,16 shots × 3 var. │   │            │                      │
│                              │   │            ▼                      │
│                              │   │  Fine-tuning (QLoRA, 1k & 2k)     │
│                              │   │            │                      │
│                              │   │            ▼                      │
│                              │   │  Meltemi / Krikri Fine-tuned eval │
└──────────────────────────────┘   └──────────────────────────────────┘
              │                                   │
              └──────────────┬────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Explanation evaluation                                           │
│    • Explanations Evaluation  — Gemini-as-judge vs. human gold refs  │
│    • Masking                  — faithfulness / flip-rate test        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository structure

| Script | Stage | Description |
|--------|-------|-------------|
| `Sampling and Machine Translation.py` | Data prep | Splits IMDb 50/50, draws balanced samples (1k & 2k train - 200 test), and translates reviews EN→EL with MADLAD-400-3B-MT. Labels are mapped to Greek (`θετικό`/`αρνητικό`). Checkpointed and resumable. |
| `Meltemi GGUF.py` | Few-shot baseline | Runs quantized **Meltemi** (Q4_K_M GGUF via `llama-cpp-python`) over `{0,2,4,6,8,16}` shots × 3 variants. Reports accuracy, per-class precision/recall/F1, and macro-F1. |
| `Llama-Krikri GGUF.py` | Few-shot baseline | Same protocol as above for **Llama-Krikri**. |
| `Training Explanations Generation.py` | Fine-tuning data prep | Uses each base model (4-bit) to generate a gold-sentiment-conditioned explanation for every training review, producing the explanation-augmented CSVs used for fine-tuning. Caches shared reviews across the 1k/2k sets. |
| `Fine-tuning.py` | Fine-tuning | **QLoRA** (4-bit NF4 + LoRA) fine-tuning of both models on both set sizes, for the three 0-shot variants. Loss is computed **only on the answer tokens**. Adapters are checkpointed and resumable. |
| `Meltemi Fine-tuned.py` | Fine-tuning evaluation | Loads the fine-tuned Meltemi LoRA adapters (1k & 2k) and evaluates sentiment accuracy + explanation length on the test set. |
| `Llama-Krikri Fine-tuned.py` | Fine-tuning evaluation | Same evaluation for the fine-tuned Llama-Krikri adapters. |
| `Explanations Evaluation.py` | Explanation quality | Uses **Gemini 2.5 Pro** as an automatic judge scoring model explanations on four criteria, then correlates the LLM scores against **human** scores (Pearson / Spearman / Kendall τ). |
| `Masking.py` | Faithfulness | Masks 50% of content words (ADJ/ADV/NOUN/VERB via Greek spaCy) in each explanation and measures how often the predicted sentiment **flips** — a proxy for whether explanations are faithful to the decision. |


---

## Evaluation dimensions

The thesis evaluates the systems along three axes:

### 1. Sentiment classification
Standard metrics computed after robust, accent-insensitive parsing of the Greek output:
- Accuracy, "unknown"/unparseable rate
- Per-class precision, recall, F1 (positive & negative)
- Macro-F1

### 2. Explanation quality (LLM-as-a-judge)
**Gemini 2.5 Pro** scores each model explanation from **1–5** on four criteria (max total = 20), given the review and a human reference explanation:

| Criterion (EL) | Meaning |
|----------------|---------|
| Ορθότητα (Correctness) | Does the explanation match the true sentiment? |
| Σύγκλιση (Alignment) | Is it similar in content/reasoning to the gold explanation? |
| Ροή (Fluency) | Is it grammatically and syntactically well-formed? |
| Πληρότητα (Completeness) | Does it fully justify why the review is positive/negative? |

Judge scores are then validated against **human ratings** using Pearson, Spearman, and Kendall τ correlations, and broken down by model, variant, and review length (short/medium/long).

### 3. Faithfulness (masking / flip-rate test)
Content words in an explanation are masked and the model is asked to re-predict sentiment **from the masked explanation alone**. A low **flip rate** (high agreement between original and masked predictions) indicates the sentiment signal is diffuse, while a high flip rate indicates the masked tokens genuinely carried the sentiment — a lightweight faithfulness probe.

---

## Requirements

Everything was developed and run on **Google Colab (GPU runtime)** with data hosted on Google Drive. Key dependencies:

```bash
# Core deep learning
pip install transformers accelerate bitsandbytes peft datasets torch

# Quantized (GGUF) inference
pip install llama-cpp-python

# Evaluation & NLP
pip install google-generativeai scikit-learn nltk scipy pandas tqdm
pip install -U spacy
python -m spacy download el_core_news_sm
```

- **GPU with CUDA** is required for 4-bit (QLoRA / bitsandbytes) inference and fine-tuning.
- A **Google Gemini API key** is required for `Explanations Evaluation.py` (set it where the code reads `genai.configure(api_key=...)`).
- Hugging Face access to the ILSP models and the MADLAD-400 model.

---

## Data

- **Source:** [IMDb 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle).
- **Split:** first 25,000 rows → training pool, remaining 25,000 → test pool (50/50).
- **Sampling:** class-balanced samples of **1k** and **2k** training reviews and a balanced **200-review** test set (100 positive + 100 negative).
- **Translation:** English → Greek with MADLAD-400-3B-MT (`<2el>` prefix).
- **Explanations:** model-generated (for fine-tuning data) and **human gold references** (for judging).

Generated artifacts referenced by the scripts include:

```
sample_training_data_{1000,2000}.csv                 # balanced, Greek-translated training sets
sample_training_data_{1000,2000}_explanations.csv    # + model-generated explanations (Fine-tuning data)
sample_test_dataset.csv                              # balanced test set
gold_references.csv                                  # human gold explanations
scored_explanations.csv / scored_explanations_human.csv
meltemi_{1000,2000}/ , llama_krikri_{1000,2000}/     # saved LoRA adapters
meltemi.gguf , llama-krikri.gguf                     # quantized model weights
```

---

## How to run

1. Place `IMDb Dataset.csv.zip` in your `MSc Thesis` Drive folder.
2. Open each script as a Colab notebook and run them in the [pipeline order](#methodology--pipeline) above.
3. For the fine-tuning track, run **Training Explanations Generation → Fine-tuning → Fine-tuned evaluation**.
4. For explanation analysis, provide `gold_references.csv` (and human scores) and add your Gemini API key before running **Explanations Evaluation**.

All long-running scripts **checkpoint to CSV** and skip already-processed rows, so they can be safely interrupted and resumed.

---

## Reproducibility notes

- Random seeds are fixed (`torch.manual_seed(42)`, `random.seed(42)`, `random_state=42`) throughout sampling, training, and generation.
- Greedy decoding (`do_sample=False`) is used for the Transformers models, GGUF inference uses low temperature (`0.1`) with a fixed seed.
- QLoRA config: 4-bit **NF4** with double quantization, LoRA `r=8`, `alpha=16`, `dropout=0.05`, targeting all attention/MLP projections, 3 epochs, LR `2e-4`, effective batch size 8 (2 × grad-accum 4).

---

## Author

**Konstantinos Kyriakopoulos** — MSc Thesis

## Acknowledgements

- **ILSP / Athena Research Center** for the [Meltemi](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5) and [Llama-Krikri](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct) Greek LLMs.
- **Google** for [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt) and the Gemini API.
- The [IMDb 50K](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset authors.

## License
